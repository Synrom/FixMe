from typing import List, Optional, Tuple
import csv
import pandas as pd
from dataclasses import asdict
from ast import literal_eval
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
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
from .tokenizer import tokenize_example
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
from ...blue import compute_blue_file, compute_blue


# pylint: disable=E1101
class UnixCoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        config,
        device,
        cutter: CutterFn,
        windowizer: WindowizerFn,
        preprocessor: PreprocessorFn,
        precutter: PreCutterFn,
        tokenizer: RobertaTokenizer,
        beam_size=None,
        sos_id=None,
        eos_id=None,
        max_length=128,
        max_source_length=256,
        max_target_length=128,
        max_window_size=2,
    ):
        super(UnixCoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024),
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device

        self.cutter = cutter
        self.windowizer = windowizer
        self.preprocessor = preprocessor
        self.precutter = precutter
        self.tokenizer = tokenizer
        self.max_token_length = 256
        self.long_tokens = {}
        for token in self.tokenizer.get_vocab().keys():
            string = self.tokenizer.convert_tokens_to_string([token])
            if len(string) > self.max_token_length:
                self.long_tokens[string] = len(string)

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_window_size = max_window_size
        self.window_length = int(self.max_target_length / 2)

    def forward(
        self, 
        source_ids = None,
        target_ids = None,
        target_token_length = None,
    ):
        if source_ids != None and target_ids != None and target_token_length != None:
            return source_ids, target_ids, target_token_length
        print("Use calc_loss or generate instead.")

    def calc_loss(
        self,
        source_ids,
        target_ids,
        target_token_length,
        fix_token_length,
    ):

        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(source_ids, attention_mask=mask, use_cache=True)
        ids = torch.cat((source_ids, target_ids), -1)
        mask = self.bias[:, source_ids.size(-1) : ids.size(-1), : ids.size(-1)].bool()
        mask = mask & ids[:, None, :].ne(1)

        out = self.decoder(
            target_ids,
            attention_mask=mask,
            past_key_values=encoder_output.past_key_values,
        )
        last_hidden_state = out.last_hidden_state
        lm_logits = self.lm_head(last_hidden_state)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        batchsize = target_token_length.shape[0]
        indices = torch.arange(self.max_target_length - 1, device=self.device).repeat(batchsize, 1)
        # mask all tokens before the fix
        # the tokens after the fix will be masked anyways since its only padding
        # This also means that the <seq> token is not masked
        indices = indices >= (target_token_length - fix_token_length - 1)
        active_loss &= indices.view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1))[active_loss],
            shift_labels.view(-1)[active_loss],
        )

        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs

    def generate(
        self,
        source_ids,  #  [batchsize, max_window, 256]
        decoder_context,  # [batchsize, 128] <cls> <decoder context + fix tokens> <sep> -> len(decoder context + fix)
        target_token_length, # [batchsize, 1]
    ):
        # embedding
        preds = []
        for batch in range(source_ids.shape[0]):
            mask = source_ids[batch].ne(1)[:, None, :] * source_ids[batch].ne(1)[:, :, None] # [max_window, 256, 256]
            encoder_output = self.encoder(source_ids[batch], attention_mask=mask, use_cache=True)
            beam = Beam(self.beam_size, self.sos_id, self.eos_id, device=self.device)
            zero = torch.LongTensor(1).fill_(0).to(self.device) # 0
            source_len = list(source_ids[batch].ne(1).sum(-1).cpu().numpy()) # [max_window] -> anzahl an nicht padding zeichen
            for window in range(source_ids.shape[1]):
                context = [
                    [
                        x[window : window + 1 ,:, : source_len[window]].repeat(self.beam_size, 1, 1, 1)
                        for x in y
                    ]
                    for y in encoder_output.past_key_values
                ] # [12, 2][10, 12, 256, 64]
                if window == 0:
                    raw_decoder_context = (
                        decoder_context[batch][: target_token_length[batch]]
                        .repeat(self.beam_size)
                        .view(self.beam_size, -1)
                    )
                    input_ids = raw_decoder_context
                else:
                    input_ids = input_ids[:, -decoder_context.shape[1] :]
                context_ids = source_ids[batch, window: window + 1, : source_len[window]].repeat(self.beam_size, 1) # [10, 256]
                for _ in range(
                    min(
                        self.max_target_length - input_ids.shape[-1],
                        decoder_context.shape[1],
                    )
                ):
                    if beam.done():
                        break
                    
                    ids = torch.cat(
                        (context_ids, input_ids), -1
                    ) # [beam_size, ]
                    mask = self.bias[
                        :, context_ids.size(-1) : ids.size(-1), : ids.size(-1)
                    ].bool()
                    mask = mask & ids[:, None, :].ne(1)
                    out = self.decoder(
                        input_ids, attention_mask=mask, past_key_values=context
                    ).last_hidden_state  # [beamsize, 1, 768]
                    hidden_states = out[:, -1, :]  # [beamsize, 768]
                    out = self.lsm(self.lm_head(hidden_states)).data  # [beamsize, 51416]
                    beam.advance(out)

                    input_ids.data.copy_(
                        input_ids.data.index_select(0, beam.getCurrentOrigin())
                    )
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
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

    def windows_train_features(self, windows: List[TokenizedExample]) -> List[InputFeaturesTrain]:
        features: List[InputFeaturesTrain] = []
        for window in windows:
            target_tokens = ["<mask0>"] + window.target_tokens
            target_token_length = len(target_tokens)
            target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = self.max_target_length - len(target_ids)
            target_ids += [self.tokenizer.pad_token_id] * padding_length
            source_window = [
                self.tokenizer.cls_token,
                "<encoder-decoder>",
                self.tokenizer.sep_token,
                "<mask0>",
            ] + window.code_tokens
            source_ids = self.tokenizer.convert_tokens_to_ids(source_window)
            padding_length = self.max_source_length - len(source_ids)
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            features.append(
                InputFeaturesTrain(
                    source_ids,
                    target_ids,
                    target_token_length,
                    window.fix_segment_length,
                    window.lang,
                )
            )
        return features

    def windows_test_features(self, windows: List[TokenizedExample]) -> List[InputFeaturesTest]:
        if len(windows) == 0:
            return []
        decoder_context = ["<mask0>"] + windows[0].target_tokens[:-windows[0].fix_segment_length]
        target_token_length = len(decoder_context)
        padding_length = self.window_length - target_token_length
        decoder_context_ids = self.tokenizer.convert_tokens_to_ids(decoder_context)
        decoder_context_ids += [self.tokenizer.pad_token_id] * padding_length
        feature = InputFeaturesTest(
            [],
            decoder_context_ids,
            target_token_length
        )
        for window in windows:
            source_window = [
                self.tokenizer.cls_token,
                "<encoder-decoder>",
                self.tokenizer.sep_token,
                "<mask0>",
            ] + window.code_tokens
            source_ids = self.tokenizer.convert_tokens_to_ids(source_window)
            padding_length = self.max_source_length - len(source_ids)
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            feature.source_ids.append(source_ids)
        return [feature]

    def tokenize_examples(self, examples: List[Example], verbose=True) -> List[TokenizedExample]:
        if verbose:
            pbar = tqdm(enumerate(examples), total=len(examples))
            pbar.set_description("tokenizing")
        else:
            pbar = enumerate(examples)
        tokenized = []
        for idx, example in pbar:
            precut = self.precutter(
                example, 
                self.max_source_length, 
                self.max_target_length, 
                self.max_token_length,
            )
            preprocess = self.preprocessor(precut)
            tokenized.append(tokenize_example(preprocess, self.tokenizer))
        return tokenized

    def tokenize_examples_encoder_as_decoder(self, examples: List[Example], verbose=True) -> List[TokenizedExample]:
        
        def equal_encoder_and_decoder_context(example: Example):
            example.target_context = example.source_until
            return example
        examples = [equal_encoder_and_decoder_context(example) for example in examples]

        if verbose:
            pbar = tqdm(enumerate(examples), total=len(examples))
            pbar.set_description("tokenizing")
        else:
            pbar = enumerate(examples)
        tokenized = []
        for idx, example in pbar:
            precut = self.precutter(
                example, 
                self.max_source_length, 
                self.max_target_length, 
                self.max_token_length,
            )
            preprocess = self.preprocessor(precut)
            tokenized.append(tokenize_example(preprocess, self.tokenizer))
        return tokenized

    def create_tokenized_dataset_decoder(self, examples: List[Example], path: str):
        tokenized = self.tokenize_examples_encoder_as_decoder(examples)
        df = pd.DataFrame(
            [
                {"tokenized": asdict(tokens)} for tokens in tokenized
            ]
        )
        df.to_csv(path, index=False)

    def create_tokenized_dataset(self, examples: List[Example], path: str):
        tokenized = self.tokenize_examples(examples)
        df = pd.DataFrame(
            [
                {"tokenized": asdict(tokens)} for tokens in tokenized
            ]
        )
        df.to_csv(path, index=False)
    
    def read_tokenized_dataset(self, path: str) -> List[TokenizedExample]:
        df = pd.read_csv(path)

        def extract_example(row) -> TokenizedExample:
            tokens = TokenizedExample(**literal_eval(row["tokenized"]))
            return tokens
        
        return df.apply(extract_example, axis=1).tolist()

    def examples_train_dataset(self, examples: List[TokenizedExample]) -> TextDatasetTrain:
        window_length = self.window_length - 1 
        max_target_length = self.max_target_length - 1 
        max_source_length = self.max_source_length - 4
        features: List[InputFeaturesTrain] = []
        pbar = tqdm(examples, total=len(examples))
        pbar.set_description("windowizing")
        for tokens in pbar:
            sample = self.cutter(tokens, max_source_length, window_length)
            windows = self.windowizer(
                sample,
                max_source_length,
                max_target_length,
                self.tokenizer.sep_token_id,
            )
            features += self.windows_train_features(windows)
        return TextDatasetTrain(
            features,
            max_window_size=self.max_window_size,
            max_source_length=self.max_source_length,
        )

    def examples_test_dataset(self, examples: List[TokenizedExample], verbose=True) -> Tuple[TextDatasetTest, List[TokenizedExample]]:
        window_length = self.window_length - 1 
        max_target_length = self.max_target_length - 1 
        max_source_length = self.max_source_length - 4
        features: List[InputFeaturesTest] = []
        counterpart_examples: List[TokenizedExample] = []
        if verbose:
            pbar = tqdm(examples, total=len(examples))
            pbar.set_description("windowizing")
        else:
            pbar = examples
        for tokens in pbar:
            sample = self.cutter(tokens, max_source_length, window_length)
            windows = self.windowizer(
                sample,
                max_source_length,
                max_target_length,
                self.tokenizer.sep_token_id,
            )
            counterpart_examples += [sample]
            features += self.windows_test_features(windows)
        return (
            TextDatasetTest(
                features,
                max_window_size=self.max_window_size,
            ),
            counterpart_examples,
        )
    
    def get_learning_rate(
        self, 
        examples: List[TokenizedExample],
        batch_size: int = 10,
    ):
        def fastai_calc_loss(forwarded, fix_token_length):
            source_ids, target_ids, target_token_length = forwarded
            return self.calc_loss(
                source_ids, 
                target_ids,
                target_token_length,
                fix_token_length
            )[0]
        dataset = self.examples_train_dataset(examples)
        dataloader = DataLoaders.from_dsets(dataset, batch_size=batch_size)
        learner = Learner(dataloader, self, fastai_calc_loss, cbs=LRFinder())
        print(learner.lr_find())

    
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
                decoder_context,
                target_token_length,
            ) = batch
            with torch.no_grad():
                preds = self.generate(
                    source_ids,
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

    def get_best_and_worst(
        self,
        testing: List[TokenizedExample],
        output_dir: str,
    ):
        device = self.device
        data = self.examples_train_dataset(testing)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data,
            sampler=sampler,
            batch_size=1,
            num_workers=1,
        )
        self.eval()
        p = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            (
                source_ids,
                target_ids,
                target_token_length,
                fix_token_length,
                lang,
            ) = batch
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)
            target_token_length = target_token_length.to(device)
            fix_token_length = fix_token_length.to(device)
            lang = lang[0]
            with torch.no_grad():
                loss, _, _ = self.calc_loss(
                    source_ids,
                    target_ids,
                    target_token_length,
                    fix_token_length,
                )
                p.append(
                    (
                        source_ids,
                        target_ids,
                        target_token_length,
                        fix_token_length,
                        loss,
                        lang,
                    )
                )
        self.train()
        results = []
        for source_ids, target_ids, target_token_length, fix_token_length, loss, lang in p:
            assert source_ids.shape[0] == 1
            assert target_ids.shape[0] == 1
            encoder_context_until = source_ids[0].tolist()
            encoder_context_from = ""
            buggy_segment = ""
            fix_token_length = fix_token_length[0][0].tolist()
            target_token_length = target_token_length[0][0].tolist()
            fixed_segment = target_ids[0].tolist()[:target_token_length][-fix_token_length:]
            decoder_context = target_ids[0].tolist()[:target_token_length][:-fix_token_length]

            def stringify(ids: List[int]):
                while self.tokenizer.pad_token_id in ids:
                    ids.remove(self.tokenizer.pad_token_id)
                string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids))
                return string

            encoder_context_until = stringify(encoder_context_until)
            fixed_segment = stringify(fixed_segment)
            decoder_context = stringify(decoder_context)
            results.append([
                encoder_context_until,
                buggy_segment,
                encoder_context_from,
                decoder_context,
                fixed_segment,
                loss.tolist(),
                lang,
            ])
        results = sorted(results, key=lambda row: row[5], reverse=True)
        
        best = os.path.join(output_dir, "best_loss.csv")
        print(f"Write best samples to {best}.")
        with open(best, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(
                [[
                    "encoder_context_until",
                    "buggy_segment",
                    "encoder_context_from",
                    "decoder_context",
                    "fixed_segment",
                    "loss",
                    "language",
                ]] + results[:100]
            )

        worst = os.path.join(output_dir, "worst_loss.csv")
        print(f"Write worst samples to {worst}.")
        with open(worst, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(
                [[
                    "encoder_context_until",
                    "buggy_segment",
                    "encoder_context_from",
                    "decoder_context",
                    "fixed_segment",
                    "loss",
                    "language",
                ]] + results[-100:]
            )



    def test_examples(
        self,
        testing: List[TokenizedExample],
        output_dir: str,
        batch_size: int = 20,
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
                decoder_context,
                target_token_length,
            ) = batch
            with torch.no_grad():
                preds = self.generate(
                    source_ids,
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
        training: List[TokenizedExample],
        output_dir: str,
        batch_size: int = 20,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5, # from code summarization
        adam_epsilon: float = 1e-8,
        epochs: int = 10,
        validation: Optional[List[TokenizedExample]] = None,
    ):
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
            inference_data, inference_examples = self.examples_test_dataset(validation)
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
                (
                    source_ids,
                    target_ids,
                    target_token_length,
                    fix_token_length,
                    _,
                ) = batch
                batch = tuple(t.to(device) for t in batch)
                source_ids = source_ids.to(device)
                target_ids = target_ids.to(device)
                target_token_length = target_token_length.to(device)
                fix_token_length = fix_token_length.to(device)
                loss, _, _ = self.calc_loss(
                    source_ids,
                    target_ids,
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
                        target_ids,
                        target_token_length,
                        fix_token_length,
                    ) = batch
                    with torch.no_grad():
                        _, loss, num = self.calc_loss(
                            source_ids,
                            target_ids,
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
            #            decoder_context,
            #            target_token_length,
            #        ) = batch
            #        with torch.no_grad():
            #            preds = self.generate(
            #                source_ids,
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
            #    print(f"  bleu-4 = {dev_bleu} ")
            #    print(f"  xMatch = {round(np.mean(accs) * 100, 4)} ")
            #    print(f" {'*' * 20}")
            #    if dev_bleu + xmatch > best_bleu:
            #        print(f"  Best BLEU+xMatch:{dev_bleu + xmatch}")
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

def build_scratch_unixcoder(
    preprocessor: PreprocessorFn = no_preprocessing,
    cutter: CutterFn = cut_context,
    windowizer: WindowizerFn = windowize,
    precutter: PreCutterFn = cut_too_long,
    beam_size: int = 10,
    max_window_size: int = 2,
    device: torch.device = None,
) -> UnixCoder:
    print("before cuda")
    if not device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(device)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")
    config.is_decoder = True
    #config.output_attentions = True
    #config.output_scores = True
    #config.output_hidden_states = True
    #config.add_cross_attention = True


    encoder = RobertaModel.from_pretrained(
        "microsoft/unixcoder-base", config=config
    )
    model = UnixCoder(
        encoder=encoder,
        decoder=encoder,
        config=config,
        device=device,
        beam_size=beam_size,
        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
        eos_id=tokenizer.sep_token_id,
        cutter=cutter,
        preprocessor=preprocessor,
        precutter=precutter,
        windowizer=windowizer,
        tokenizer=tokenizer,
        max_window_size=max_window_size,
    )

    print("before to device")
    model.to(device)
    print("after to device")
    return model

