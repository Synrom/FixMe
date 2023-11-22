print("start import")
from typing import List, Optional, Tuple
from ast import literal_eval
import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import pandas as pd
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaConfig,
    RobertaModel,
)
from fastai.learner import Learner
from fastai.callback.schedule import LRFinder
from fastai.text.data import DataLoaders
from tqdm import tqdm
from ...beam import Beam
from .tokenizer import tokenize_example, DFG, ORI2CUR
from .dataset import (
    TextDatasetTrain,
    InputFeaturesTrain,
    InputFeaturesTest,
    TextDatasetTest,
    TestSample,
    TrainSample,
)
from ...cutter import CutterFn, cut_context
from ...windowizer import WindowizerFn, windowize
from ...preprocessor import PreprocessorFn, no_preprocessing
from ...precutter import PreCutterFn, cut_too_long
from ...example import TokenizedExample, Example
from ...blue import compute_blue_file


# pylint: disable=E1101
class GraphCodeBERT(nn.Module):
    """
    Build Seqence-to-Sequence.

    Parameters:

    * `encoder`- encoder of seq2seq model. e.g. roberta
    * `decoder`- decoder of seq2seq model. e.g. transformer
    * `config`- configuration of encoder model.
    * `beam_size`- beam size for beam search.
    * `sos_id`- start of symbol ids in target for beam search.
    * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(
        self,
        encoder,
        decoder,
        config,
        device: torch.device,
        cutter: CutterFn,
        windowizer: WindowizerFn,
        preprocessor: PreprocessorFn,
        precutter: PreCutterFn,
        beam_size=None,
        sos_id=None,
        eos_id=None,
        max_source_length=320,
        max_target_length=256,
        max_window_size=2,
    ):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device
        self.max_window_size = max_window_size
        tokenizer_class = RobertaTokenizer
        self.tokenizer = tokenizer_class.from_pretrained("microsoft/graphcodebert-base")
        self.max_token_length = max(len(self.tokenizer.convert_tokens_to_string([token])) for token in self.tokenizer.get_vocab().keys())
        print(f"max_token_length is {self.max_token_length}")
        self.cutter = cutter
        self.windowizer = windowizer
        self.preprocessor = preprocessor
        self.precutter = precutter
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.window_length = int(self.max_target_length / 2)

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using TorchScript or not"""
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.lm_head, self.encoder.embeddings.word_embeddings
        )

    def forward(
        self,
        source_ids=None,
        source_mask=None,
        position_idx=None,
        attn_mask=None,
        target_ids=None,
        target_mask=None,
        target_token_length=None,
    ):
        return source_ids, source_mask, position_idx, attn_mask, target_ids, target_mask, target_token_length
        print("Use calc_loss or generate instead.")

    def calc_loss(
        self,
        source_ids,  # [batchsize, 320] | pred: [batchsize, max_window, 320]
        source_mask,  # [batchsize, 320] | pred: [batchsize, max_window, 320]
        position_idx,  # [batchsize, 320] | pred: [batchsize, max_window, 320]
        attn_mask,  # [batchsize, 320, 320] the dfg_to_dfg etc stuff | pred: [batchsize, max_window, 320, 320]
        target_ids,  # [batchsize, 256] includes padding at end
        target_mask,  # [batchsize, 256] defines the target padding
        target_token_length,  # [batchsize, 1] <cls> <decoder context + fix tokens> <sep> -> len(decoder context + fix)
        fix_token_length,  # [batchsize, 1] len(fix tokens)
    ):
        # embedding
        # position_idx == 0 -> a dfg variable
        nodes_mask = position_idx.eq(0)  # [1, 320]

        # position_idx >= 2 -> a normal token
        token_mask = position_idx.ge(2)  # [1, 320]
        inputs_embeddings = self.encoder.embeddings.word_embeddings(
            source_ids
        )  # [1, 320, 768]
        nodes_to_token_mask = (
            nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        )
        nodes_to_token_mask = (
            nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        )  # [1, 320, 320]
        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
        )  # [1, 320, 768]
        inputs_embeddings = (
            inputs_embeddings
            * (~nodes_mask)[
                :, :, None
            ]  # get the sum of the token variables as embedding for the variable
            + avg_embeddings * nodes_mask[:, :, None]
        )  # [1, 320, 768]

        outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )  # [1, 320, 768]
        encoder_output = (
            outputs[0].permute([1, 0, 2]).contiguous()
        )  # [320, batchsize, 768]
        # source_mask=token_mask.float()
        # attn_mask will be (0, -1, -1 ...) (0, 0 , -1 ...) ...
        attn_mask = -1e4 * (
            1 - self.bias[: target_ids.shape[1], : target_ids.shape[1]]
        )  # [256, 256]
        tgt_embeddings = (
            self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
        )  # [256, batchsize, 768]
        out = self.decoder(
            tgt_embeddings,
            encoder_output,
            tgt_mask=attn_mask,
            memory_key_padding_mask=(1 - source_mask).bool(),
        )  # [256, 1, 768]
        hidden_states = (
            torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
        )  # [1, 256, 768]
        lm_logits = self.lm_head(hidden_states)  # [1, 256, 50265] so vocabulary size
        # Shift so that tokens < n predict n; target_mask is 1 for tokens and 0 for padding
        active_loss = (
            target_mask[..., 1:].ne(0).view(-1) == 1
        )  # [255 * batchsize], True only for tokens
        batchsize = target_token_length.shape[0]
        indices = torch.arange(255, device=self.device).repeat(batchsize, 1)
        # mask all tokens before the fix
        # the tokens after the fix will be masked anyways since its only padding
        indices = indices >= target_token_length - fix_token_length - 1
        active_loss &= indices.view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()  # [1, 256, 50265]
        shift_labels = target_ids[
            ..., 1:
        ].contiguous()  # [1, 256] the real target tokens
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1))[
                active_loss
            ],  # [37, 50265] -> because active loss is True only 37 times
            shift_labels.view(-1)[
                active_loss
            ],  # [37] containing token labels from 0 to 50265
        )

        outputs = (
            loss,
            loss * active_loss.sum(),
            active_loss.sum(),
        )  # avg loss, summed loss, number of predicted tokens
        return outputs

    def generate(
        self,
        source_ids,  #  [batchsize, max_window, 320]
        source_mask,  # [batchsize, max_window, 320]
        position_idx,  # [batchsize, max_window, 320]
        attn_mask,  # [batchsize, max_window, 320, 320]
        decoder_context,  # [batchsize, 1] <cls> <decoder context + fix tokens> <sep> -> len(decoder context + fix)
        target_token_length,
    ):
        # embedding
        preds = []
        for batch in range(source_ids.shape[0]):
            # position_idx == 0 -> a dfg variable
            nodes_mask = position_idx[batch].eq(0)  # [4, 320]

            # position_idx >= 2 -> a normal token
            token_mask = position_idx[batch].ge(2)  # [4, 320]
            inputs_embeddings = self.encoder.embeddings.word_embeddings(
                source_ids[batch]
            )  # [4, 320, 768]
            nodes_to_token_mask = (
                nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask[batch]
            )
            nodes_to_token_mask = (
                nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            )  # [4, 320, 320]
            avg_embeddings = torch.einsum(
                "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
            )  # [1, 320, 768]
            inputs_embeddings = (
                inputs_embeddings
                * (~nodes_mask)[
                    :, :, None
                ]  # get the sum of the token variables as embedding for the variable
                + avg_embeddings * nodes_mask[:, :, None]
            )  # [4, 320, 768]

            outputs = self.encoder(
                inputs_embeds=inputs_embeddings,
                attention_mask=attn_mask[batch],
                position_ids=position_idx[batch],
            )  # [4, 320, 768]
            encoder_output = outputs[0].permute([1, 0, 2]).contiguous()  # [320, 4, 768]
            # source_mask=token_mask.float()
            # Predict with decoder context
            beam = Beam(self.beam_size, self.sos_id, self.eos_id, device=self.device)
            for window in range(source_ids.shape[1]):
                zero = torch.LongTensor(1).fill_(0).to(self.device)
                context = encoder_output[:, window : window + 1]
                context_mask = source_mask[batch, window : window + 1, :]
                if window == 0:
                    raw_decoder_context = (
                        decoder_context[batch][: target_token_length[batch]]
                        .repeat(self.beam_size)
                        .view(self.beam_size, -1)
                    )
                    input_ids = raw_decoder_context
                else:
                    input_ids = input_ids[:, -decoder_context.shape[1] :]
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(
                    min(
                        self.max_target_length - input_ids.shape[-1],
                        decoder_context.shape[1],
                    )
                ):
                    if beam.done():
                        break
                    tgt_mask = -1e4 * (
                        1 - self.bias[: input_ids.shape[1], : input_ids.shape[1]]
                    )
                    tgt_embeddings = (
                        self.encoder.embeddings(input_ids)
                        .permute([1, 0, 2])
                        .contiguous()
                    )
                    out = self.decoder(
                        tgt_embeddings,
                        context,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=(1 - context_mask).bool(),
                    )
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(
                        input_ids.data.index_select(0, beam.getCurrentOrigin())
                    )
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                # hyp = beam.getHyp(beam.getFinal())
                # pred = beam.buildTargetTokens(hyp)[: self.beam_size]
                # pred = [
                #     torch.cat(
                #         [x.view(-1) for x in p]
                #         + [zero] * (decoder_context.shape[1] * source_ids.shape[1] - len(p))
                #     ).view(1, -1)
                #     for p in pred
                # ]
                # pred = torch.cat(pred, 0)
                # t = pred[0].cpu().numpy()
                # t = list(t)
                # if (0 in t).any():
                #     t = t[: t.index(0)]
                # text = self.tokenizer.decode(
                #     t, clean_up_tokenization_spaces=False
                # )
                # print(f"in window {window}:")
                # print(text)
                    
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[: self.beam_size]
            pred = [
                torch.cat(
                    [x.view(-1) for x in p]
                    + [zero] * (decoder_context.shape[1] * source_ids.shape[1] - len(p))
                ).view(1, -1)
                for p in pred
            ]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)
        return preds

    def windows_train_features(
        self, windows: List[TokenizedExample], dfg: DFG, ori2cur_pos: ORI2CUR
    ) -> List[InputFeaturesTrain]:
        features: List[InputFeaturesTrain] = []
        for example in windows:
            # target

            target_token_length = len(example.target_tokens)
            target_ids = self.tokenizer.convert_tokens_to_ids(example.target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = self.max_target_length - len(target_ids)
            target_ids += [self.tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length

            # source
            source_tokens = example.code_tokens
            source_ids = self.tokenizer.convert_tokens_to_ids(example.code_tokens)
            position_idx = [
                i + self.tokenizer.pad_token_id + 1 for i in range(len(source_ids))
            ]
            dfg = dfg[: max(self.max_source_length - len(source_ids), 0)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [self.tokenizer.unk_token_id for x in dfg]
            padding_length = self.max_source_length - len(source_ids)
            position_idx += [self.tokenizer.pad_token_id] * padding_length
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            source_mask = [1] * (len(source_tokens))
            source_mask += [0] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1] + (
                    [reverse_index[i] for i in x[-1] if i in reverse_index],
                )
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([self.tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

            features.append(
                InputFeaturesTrain(
                    source_ids,
                    position_idx,
                    dfg_to_dfg,
                    dfg_to_code,
                    target_ids,
                    source_mask,
                    target_mask,
                    target_token_length,
                    example.fix_segment_length,
                )
            )
        return features

    def predict_one_example(
        self,
        example: Example,
    ) -> str:
        device = self.device
        tokenized = self.tokenize_examples([example], verbose=False)
        data, _ = self.examples_test_dataset(tokenized, verbose=False)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data,
            sampler=sampler,
        )
        self.eval()
        p = []
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            (
                source_ids,
                source_mask,
                positional_idx,
                attn_mask,
                decoder_context,
                target_token_length,
            ) = batch
            with torch.no_grad():
                preds = self.generate(
                    source_ids,
                    source_mask,
                    positional_idx,
                    attn_mask,
                    decoder_context,
                    target_token_length,
                )
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text = self.tokenizer.decode(
                        t, clean_up_tokenization_spaces=False
                    )
                    p.append(text)
        self.train()
        assert len(p) == 1
        return p[0]

    def windows_test_features(
        self, windows: List[TokenizedExample], dfg: DFG, ori2cur_pos: ORI2CUR
    ) -> List[InputFeaturesTest]:
        if len(windows) == 0:
            return []
        decoder_context = windows[0].target_tokens
        decoder_context = decoder_context[
            : -windows[0].fix_segment_length
        ]
        decoder_context_ids = self.tokenizer.convert_tokens_to_ids(decoder_context)
        target_length = len(decoder_context_ids)
        target_mask = [1] * target_length
        padding_length = self.window_length - target_length
        decoder_context_ids += [self.tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        feature = InputFeaturesTest(
            [],
            [],
            [],
            [],
            [],
            decoder_context_ids,
            target_length,
        )
        for example in windows:
            # source
            source_tokens = example.code_tokens
            source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [
                i + self.tokenizer.pad_token_id + 1 for i in range(len(source_tokens))
            ]
            dfg = dfg[: max(self.max_source_length - len(source_tokens), 0)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [self.tokenizer.unk_token_id for x in dfg]
            padding_length = self.max_source_length - len(source_ids)
            position_idx += [self.tokenizer.pad_token_id] * padding_length
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            source_mask = [1] * (len(source_tokens))
            source_mask += [0] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1] + (
                    [reverse_index[i] for i in x[-1] if i in reverse_index],
                )
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([self.tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

            feature.dfg_to_code.append(dfg_to_code)
            feature.dfg_to_dfg.append(dfg_to_dfg)
            feature.source_ids.append(source_ids)
            feature.position_idx.append(position_idx)
            feature.source_mask.append(source_mask)
        return [feature]

    def tokenize_examples(self, examples: List[Example], verbose=True) -> List[Tuple[TokenizedExample, DFG, ORI2CUR]]:
        if verbose:
            pbar = tqdm(enumerate(examples), total=len(examples))
            pbar.set_description("tokenization")
        else:
            pbar = enumerate(examples)
        return [
            tokenize_example(
                self.preprocessor(
                    self.precutter(
                        example,
                        self.max_source_length,
                        self.max_target_length,
                        self.max_token_length,
                    ), 
                ),
                self.tokenizer, idx
            )
            for idx, example in pbar
        ]

    def create_tokenized_dataset(self, examples: List[Example], path: str):
        tokenized = self.tokenize_examples(examples)
        df = pd.DataFrame(
            [
                {
                    "tokenized": asdict(tokens),
                    "dfg": dfg,
                    "ori2cur": ori2cur,
                }
                for tokens, dfg, ori2cur in tokenized
            ]
        )
        df.to_csv(path, index=False)
    
    def read_tokenized_dataset(self, path: str) -> List[Tuple[TokenizedExample, DFG, ORI2CUR]]:
        df = pd.read_csv(path)

        def extract_example(row) -> Tuple[TokenizedExample, DFG, ORI2CUR]:
            tokens = TokenizedExample(**literal_eval(row["tokenized"]))
            dfg = literal_eval(row["dfg"])
            ori2cur = literal_eval(row["ori2cur"])
            return tokens, dfg, ori2cur
        
        examples = df.apply(extract_example, axis=1).tolist()
        minimum = None
        maximum = None
        for example, _, _ in examples:
            length = min(len(example.code_tokens), len(example.target_tokens))
            if not minimum or length < minimum:
                minimum = length
            length = max(len(example.code_tokens), len(example.target_tokens))
            if not maximum or length > maximum:
                maximum = length
        print(f"Minimum Token length is {minimum}.")
        print(f"Maximum Token length is {maximum}.")
        return examples


    def examples_train_dataset(self, examples: List[Tuple[TokenizedExample, DFG, ORI2CUR]]) -> TextDatasetTrain:
        features: List[InputFeaturesTrain] = []
        pbar = tqdm(examples, total=len(examples))
        pbar.set_description("windowizing")
        for tokens, dfg, ori2cur in pbar:
            sample = self.cutter(tokens, self.max_source_length, self.window_length)
            windows = self.windowizer(
                sample,
                self.max_source_length,
                self.max_target_length,
                self.tokenizer.sep_token_id,
            )
            features += self.windows_train_features(windows, dfg, ori2cur)
        return TextDatasetTrain(
            features,
            max_window_size=self.max_window_size,
            max_source_length=self.max_source_length,
        )

    def examples_test_dataset(
        self, examples: List[Tuple[TokenizedExample, DFG, ORI2CUR]], verbose=True,
    ) -> Tuple[TextDatasetTest, List[TokenizedExample]]:
        features: List[InputFeaturesTest] = []
        counterpart_examples: List[TokenizedExample] = []
        if verbose:
            pbar = tqdm(examples, total=len(examples), desc="windowizing")
            pbar.set_description("windowizing")
        else:
            pbar = examples
        for tokens, dfg, ori2cur in pbar:
            sample = self.cutter(tokens, self.max_source_length, self.window_length)
            windows = self.windowizer(
                sample,
                self.max_source_length,
                self.max_target_length,
                self.tokenizer.sep_token_id,
            )
            counterpart_examples += [sample]
            features += self.windows_test_features(windows, dfg, ori2cur)
        return (
            TextDatasetTest(
                features,
                max_window_size=self.max_window_size,
                max_source_length=self.max_source_length,
            ),
            counterpart_examples,
        )

    def get_learning_rate(
        self, 
        examples: List[Tuple[TokenizedExample, DFG, ORI2CUR]],
        batch_size: int = 10,
    ):
        def fastai_calc_loss(forwarded, fix_token_length):
            source_ids, source_mask, position_idx, attn_mask, target_ids, target_mask, target_token_length = forwarded
            return self.calc_loss(
                source_ids, 
                source_mask,
                position_idx,
                attn_mask,
                target_ids,
                target_mask,
                target_token_length,
                fix_token_length
            )[0]
        dataset = self.examples_train_dataset(examples)
        dataloader = DataLoaders.from_dsets(dataset, batch_size=batch_size)
        learner = Learner(dataloader, self, fastai_calc_loss, cbs=LRFinder())
        print(learner.lr_find())

    def test_examples(
        self,
        testing: List[Tuple[TokenizedExample, DFG, ORI2CUR]],
        output_dir: str,
        batch_size: int = 10,
    ):
        device = self.device
        data, examples = self.examples_test_dataset(testing)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
        )
        self.eval()
        p = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            (
                source_ids,
                source_mask,
                position_idx,
                attn_mask,
                decoder_context,
                target_token_length,
            ) = batch
            with torch.no_grad():
                preds = self.generate(
                    source_ids,
                    source_mask,
                    position_idx,
                    attn_mask,
                    decoder_context,
                    target_token_length,
                )
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text = self.tokenizer.decode(
                        t, clean_up_tokenization_spaces=False
                    )
                    p.append(text)
        self.train()
        accs = []
        assert len(p) == len(examples)
        with open(
            os.path.join(output_dir, "dev.output"),
            "w",
        ) as f, open(
            os.path.join(output_dir, "dev.gold"),
            "w",
        ) as f1:
            for ref, gold in zip(p, examples):
                f.write(ref.replace("\n", "\\n") + "\n")
                correct = self.tokenizer.convert_tokens_to_string(gold.target_tokens[-gold.fix_segment_length :][:-1])
                f1.write(correct.replace("\n", "\\n") + "\n")
                accs.append(ref == correct)

        dev_bleu = round(
            compute_blue_file(
                os.path.join(output_dir, "dev.gold"),
                os.path.join(output_dir, "dev.output"),
            ),
            2,
        )
        xmatch = round(np.mean(accs) * 100, 4)
        print(f"  bleu-4 = {dev_bleu} ")
        print(f"  xMatch = {round(np.mean(accs) * 100, 4)} ")

    def train_examples(
        self,
        training: List[Tuple[TokenizedExample, DFG, ORI2CUR]],
        output_dir: str,
        batch_size: int = 15,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4, # from code translation
        adam_epsilon: float = 1e-8,
        epochs: int = 20,
        validation: Optional[List[Tuple[TokenizedExample, DFG, ORI2CUR]]] = None,
    ):
        print(f"train for {epochs} epochs.")
        batch_size = 10
        device = self.device
        dataset = self.examples_train_dataset(training)
        sampler = RandomSampler(dataset)
        dataloader: DataLoader[TrainSample] = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size // gradient_accumulation_steps,
            num_workers=4,
        )
        known_eval_dataloader: Optional[DataLoader[TestSample]] = None
        if validation:
            known_eval_data = self.examples_train_dataset(validation)
            known_eval_sampler = SequentialSampler(known_eval_data)
            known_eval_dataloader = DataLoader(
                known_eval_data,
                sampler=known_eval_sampler,
                batch_size=batch_size,
                num_workers=4,
            )
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(dataloader) * epochs * 0.1,
            num_training_steps=len(dataloader) * epochs,
        )

        # Start training
        print("***** Running training *****")
        print(f"  Num examples = {len(training)}")
        print(f"  Batch size = {batch_size}")
        print(f"  Num epoch = {epochs}")

        self.train()
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = (
            0,
            0,
            0,
            0,
            0,
            1e6,
        )
        for epoch in range(epochs):
            bar = tqdm(dataloader, total=len(dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                (
                    source_ids,
                    source_mask,
                    position_idx,
                    att_mask,
                    target_ids,
                    target_mask,
                    target_token_length,
                    fix_token_length,
                ) = batch
                loss, _, _ = self.calc_loss(
                    source_ids,
                    source_mask,
                    position_idx,
                    att_mask,
                    target_ids,
                    target_mask,
                    target_token_length,
                    fix_token_length,
                )

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss = round(
                    tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4
                )
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            # Eval model with dev dataset
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            print("\n***** Running evaluation *****")
            print(f"  Num examples = {len(known_eval_data)}")

            # Start Evaling model
            self.eval()
            eval_loss: float = 0
            tokens_num: int = 0
            if known_eval_dataloader:
                for batch in known_eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    (
                        source_ids,
                        source_mask,
                        position_idx,
                        att_mask,
                        target_ids,
                        target_mask,
                        target_token_length,
                        fix_token_length,
                    ) = batch
                    with torch.no_grad():
                        _, loss, num = self.calc_loss(
                            source_ids,
                            source_mask,
                            position_idx,
                            att_mask,
                            target_ids,
                            target_mask,
                            target_token_length,
                            fix_token_length,
                        )
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                self.train()
                eval_loss = eval_loss / tokens_num
                result = {
                    "eval_ppl": round(np.exp(eval_loss), 5),
                    "global_step": global_step + 1,
                    "train_loss": round(train_loss, 5),
                }
                for key in sorted(result.keys()):
                    print(f"  {key} = {result[key]}")

                # save last checkpoint
                last_output_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = (
                    self.module if hasattr(self, "module") else self
                )  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    print(f"  Best ppl:{round(np.exp(eval_loss), 5)}")
                    print(f" {'*' * 20}")
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    best_output_dir = os.path.join(output_dir, "checkpoint-best-ppl")
                    if not os.path.exists(best_output_dir):
                        os.makedirs(best_output_dir)
                    model_to_save = (
                        self.module if hasattr(self, "module") else self
                    )  # Only save the model it-self
                    output_model_file = os.path.join(best_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

            #if inference_dataloader:
            #    self.eval()
            #    p = []
            #    for batch in inference_dataloader:
            #        batch = tuple(t.to(device) for t in batch)
            #        (
            #            source_ids,
            #            source_mask,
            #            position_idx,
            #            att_mask,
            #            decoder_context,
            #            target_token_length,
            #        ) = batch
            #        with torch.no_grad():
            #            preds = self.generate(
            #                source_ids,
            #                source_mask,
            #                position_idx,
            #                att_mask,
            #                decoder_context,
            #                target_token_length,
            #            )
            #            for pred in preds:
            #                t = pred[0].cpu().numpy()
            #                t = list(t)
            #                if 0 in t:
            #                    t = t[: t.index(0)]
            #                text = self.tokenizer.decode(
            #                    t, clean_up_tokenization_spaces=False
            #                )
            #                p.append(text)
            #    #for pred in p:
            #    #    print("prediction:::::::::::")
            #    #    print(pred)
            #    #print(p)
            #    
            #    self.train()
            #    predictions = []
            #    accs = []
            #    with open(
            #        os.path.join(output_dir, f"checkpoint-{epoch}", "dev.output"),
            #        "w",
            #    ) as f, open(
            #        os.path.join(output_dir, f"checkpoint-{epoch}", "dev.gold"),
            #        "w",
            #    ) as f1:
            #        for ref, gold in zip(p, inference_examples):
            #            predictions.append(ref)
            #            f.write(ref.replace("\n", "\\n") + "\n")
            #            correct = self.tokenizer.decode(
            #                self.tokenizer.convert_tokens_to_ids(
            #                    gold.target_tokens[-gold.fix_segment_length :]
            #                ),
            #                clean_up_tokenization_spaces=False,
            #            )
            #            f1.write(correct.replace("\n", "\\n") + "\n")
            #            accs.append(ref == correct)

            #    dev_bleu = round(
            #        compute_blue_file(
            #            os.path.join(output_dir, f"checkpoint-{epoch}", "dev.gold"),
            #            os.path.join(output_dir, f"checkpoint-{epoch}", "dev.output"),
            #        ),
            #        2,
            #    )
            #    xmatch = round(np.mean(accs) * 100, 4)
            #    print(f"  bleu-4 = {dev_bleu} " % ("bleu-4", str(dev_bleu)))
            #    print(f"  xMatch = {round(np.mean(accs) * 100, 4)} ")
            #    print(f" {'*' * 20}")
            #    if dev_bleu + xmatch > best_bleu:
            #        print(f"  Best BLEU+xMatch:{dev_bleu+xmatch}")
            #        print(f" {'*' * 20}")
            #        best_bleu = dev_bleu + xmatch
            #        # Save best checkpoint for best bleu
            #        best_output_dir = os.path.join(output_dir, "checkpoint-best-bleu")
            #        if not os.path.exists(best_output_dir):
            #            os.makedirs(best_output_dir)
            #        model_to_save = (
            #            self.module if hasattr(self, "module") else self
            #        )  # Only save the model it-self
            #        output_model_file = os.path.join(best_output_dir, "pytorch_model.bin")
            #        torch.save(model_to_save.state_dict(), output_model_file)

def build_scratch_graphcodebert(
    preprocessor: PreprocessorFn = no_preprocessing,
    cutter: CutterFn = cut_context,
    windowizer: WindowizerFn = windowize,
    precutter: PreCutterFn = cut_too_long,
    beam_size: int = 10,
    max_window_size: int = 2,
    device: torch.device = None, 
) -> GraphCodeBERT:
    if not device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(device)
    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    encoder = RobertaModel.from_pretrained(
        "microsoft/graphcodebert-base", config=config
    )
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = GraphCodeBERT(
        encoder=encoder,
        decoder=decoder,
        config=config,
        device=device,
        beam_size=beam_size,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
        preprocessor=preprocessor,
        precutter=precutter,
        cutter=cutter,
        windowizer=windowizer,
        max_window_size=max_window_size,
    )

    model.to(device)
    return model
