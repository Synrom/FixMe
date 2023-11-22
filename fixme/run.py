from typing import List, Union
import time
import cchardet
import pandas as pd
from tqdm import tqdm
import os
import sys
import torch
from .example import Example
from .preprocessor import delete_indentation, no_context
from .models.graphcodebert.model import build_scratch_graphcodebert, GraphCodeBERT
from .models.unixcoder.model import build_scratch_unixcoder, UnixCoder
from .precutter import no_cutter
from .windowizer import no_windowization
import logging



def try_model(model: Union[GraphCodeBERT, UnixCoder]):
    encoder_context_before = """        public void test_for_issue() throws Exception {
        LoginResponse loginResp = new LoginResponse();
"""

    buggy_segment = """     loginResp.response = new Response<LoginResponse.Body>();
        loginResp.response.content = new Body();
        loginResp.response.content.setMemberinfo(new MemberInfo());
        loginResp.response.content.getMemberinfo().name = "ding102992";
        loginResp.response.content.getMemberinfo().email = "ding102992@github.com";
"""

    encoder_context_after = """

        String text = JSON.toJSONString(loginResp);

        LoginResponse loginResp2 = JSON.parseObject(text, LoginResponse.class);
        
        Assert.assertEquals(loginResp.response //
                                     .getContent() //
                                     .getMemberinfo().name, //
                            loginResp2.response //
                                      .getContent() //
                                      .getMemberinfo().name);
        Assert.assertEquals(loginResp.response //
                                     .getContent().getMemberinfo().email, //
                            loginResp2.response.getContent().getMemberinfo().email);

    }"""

    decoder_context = """   public void test_for_issue() throws Exception {
        LoginResponse loginResp = new LoginResponse();
"""

    fixed_segment = """     loginResp.setResponse(new Response<LoginResponse.Body>());
        loginResp.getResponse().setContent(new Body());
        loginResp.getResponse().getContent().setMemberinfo(new MemberInfo());
        loginResp.getResponse().getContent().getMemberinfo().name = "ding102992";
        loginResp.getResponse().getContent().getMemberinfo().email = "ding102992@github.com";"""
    samples = [
        Example(
            encoder_context_before,
            buggy_segment,
            encoder_context_after,
            fixed_segment,
            decoder_context,
            lang="Java",
        )
    ]
    encoder_context_before = """    public List<Request> getRequests(UUID taskuuid) {
        List<Request> list = new ArrayList<Request>();
        try {
            if (taskuuid != null) {
                String cql = "select * from requests WHERE taskuuid = ? order by agentuuid, reqseqnum;";
                PreparedStatement stmt = session.prepare(cql);
                BoundStatement boundStatement = new BoundStatement(stmt);
                ResultSet rs = session.execute(boundStatement.bind(taskuuid));
"""
    buggy_segment = """                for (Row row : rs) {
                    Request request = new Request();
                    request.setProgram(row.getString("program"));
                    request.setArgs(row.getList("args", String.class));
                    request.setEnvironment(row.getMap("environment", String.class, String.class));
                    request.setPid(row.getInt("pid"));
                    request.setProgram(row.getString("program"));
                    request.setRequestSequenceNumber(row.getInt("reqseqnum"));
                    request.setRunAs(row.getString("runsas"));
                    request.setSource(row.getString("source"));
                    if (row.getString("outputredirectionstderr") != null) {
                        request.setStdErr(OutputRedirection.valueOf(row.getString("outputredirectionstderr")));
                    }
                    request.setStdErrPath(row.getString("erroutpath"));
                    if (row.getString("outputredirectionstdout") != null) {
                        request.setStdOut(OutputRedirection.valueOf(row.getString("outputredirectionstdout")));
                    }
                    request.setStdOutPath(row.getString("stdoutpath"));
                    request.setTaskUuid(row.getUUID("taskuuid"));
                    request.setTimeout(row.getInt("timeout"));
                    request.setType(RequestType.valueOf(row.getString("type")));
                    request.setUuid(row.getUUID("agentuuid"));
                    request.setWorkingDirectory(row.getString("workingdirectory"));
                    list.add(request);
"""
    encoder_context_after = """                }
            }

        } catch (Exception ex) {
            LOG.log(Level.SEVERE, "Error in getRequests", ex);
        }
        return list;
    }"""
    decoder_context = """    public List<Request> getRequests(UUID taskuuid) {
        List<Request> list = new ArrayList<Request>();
        try {
            String cql = "select * from requests";
            ResultSet rs;
            if (taskuuid == null) {
                cql += " order by agentuuid, reqseqnum;";
                rs = session.execute(cql);
            } else {
                cql += " WHERE taskuuid = ? order by agentuuid, reqseqnum;";
                PreparedStatement stmt = session.prepare(cql);

                BoundStatement boundStatement = new BoundStatement(stmt);
                rs = session.execute(boundStatement.bind(taskuuid));
            }
"""
    fixed_segment = """            for (Row row : rs) {
                Request request = new Request();
                request.setProgram(row.getString("program"));
                request.setArgs(row.getList("args", String.class));
                request.setEnvironment(row.getMap("environment", String.class, String.class));
                request.setPid(row.getInt("pid"));
                request.setProgram(row.getString("program"));
                request.setRequestSequenceNumber(row.getInt("reqseqnum"));
                request.setRunAs(row.getString("runsas"));
                request.setSource(row.getString("source"));
                if (row.getString("outputredirectionstderr") != null) {
                    request.setStdErr(OutputRedirection.valueOf(row.getString("outputredirectionstderr")));"""
    samples.append(
        Example(
            encoder_context_before,
            buggy_segment,
            encoder_context_after,
            fixed_segment,
            decoder_context,
            lang="Java",
        )
    )
    encoder_context_before = '\tpublic void shatter( int cell ) {\n\t\t\n\t\tPathFinder.buildDistanceMap( cell, BArray.not( Dungeon.level.solid, null ), DISTANCE );\n\t\t\n\t\tArrayList<Blob> blobs = new ArrayList<>();\n\t\tfor (Class c : affectedBlobs){\n\t\t\tBlob b = Dungeon.level.blobs.get(c);\n\t\t\tif (b != null && b.volume > 0){\n\t\t\t\tblobs.add(b);\n\t\t\t}\n\t\t}\n\t\t\n\t\tfor (int i=0; i < Dungeon.level.length(); i++) {\n\t\t\tif (PathFinder.distance[i] < Integer.MAX_VALUE) {\n\t\t\t\t\n\t\t\t\tfor (Blob blob : blobs) {\n'
    buggy_segment = '\t\t\t\t\tblob.clear(i);\n'
    encoder_context_after = '\t\t\t\t}\n\t\t\t\t\n\t\t\t\tif (Dungeon.level.heroFOV[i]) {\n\t\t\t\t\tCellEmitter.get( i ).burst( Speck.factory( Speck.DISCOVER ), 2 );\n\t\t\t\t}\n\t\t\t\t\n\t\t\t}\n\t\t}\n\t\t\n\t\t\n\t\tif (Dungeon.level.heroFOV[cell]) {\n\t\t\tsplash(cell);'
    decoder_context = '\tpublic void shatter( int cell ) {\n\t\t\n\t\tPathFinder.buildDistanceMap( cell, BArray.not( Dungeon.level.solid, null ), DISTANCE );\n\t\t\n\t\tArrayList<Blob> blobs = new ArrayList<>();\n\t\tfor (Class c : affectedBlobs){\n\t\t\tBlob b = Dungeon.level.blobs.get(c);\n\t\t\tif (b != null && b.volume > 0){\n\t\t\t\tblobs.add(b);\n\t\t\t}\n\t\t}\n\t\t\n\t\tfor (int i=0; i < Dungeon.level.length(); i++) {\n\t\t\tif (PathFinder.distance[i] < Integer.MAX_VALUE) {\n\t\t\t\t\n\t\t\t\tfor (Blob blob : blobs) {\n'
    fixed_segment = '\n\t\t\t\t\tint value = blob.cur[i];\n\t\t\t\t\tif (value > 0) {\n\t\t\t\t\t\tblob.clear(i);\n\t\t\t\t\t\tblob.cur[i] = 0;\n\t\t\t\t\t\tblob.volume -= value;\n\t\t\t\t\t}\n'
    samples = [
        Example(
            encoder_context_before,
            buggy_segment,
            encoder_context_after,
            fixed_segment,
            decoder_context,
            lang="Java",
        )
    ]
    tokenized_samples = model.tokenize_examples(samples)
    if not os.path.exists("try_output"):
        os.mkdir("try_output")
    model.train_examples(tokenized_samples, "try_output", validation=tokenized_samples, batch_size=1)

def predict_defects4j(source_path: str, target_path: str, model: UnixCoder):
    df = pd.read_csv(source_path,
        dtype = {
            "encoder_context_before": "string",
            "buggy_segment": "string",
            "encoder_context_from": "string",
            "decoder_context": "string",
            "fixed_segment": "string",
            "start_lineno": "int",
            "end_lineno": "int",
            "pid": "string",
            "bid": "string",
            "buggy_path": "string",
            "fixed_path": "string",
        }
    )

    pbar = tqdm(total=len(df))
    pbar.set_description("Predict")
    def predict_row(row):
        nonlocal pbar
        example = Example(
            row["encoder_context_before"],
            row["buggy_segment"],
            row["encoder_context_from"],
            "",
            row["decoder_context"],
            "Java",
        )
        prediction =  model.predict_one_example(example)
        pbar.update()
        return prediction

    start_time = time.time()
    df["prediction"] = df.apply(predict_row, axis=1)
    end_time = time.time()

    pbar.close()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    df.to_csv(target_path, index=False)

def read_examples_bugs2fix(filename) -> List[Example]:
    """Read examples from filename."""
    examples = []
    source, target = filename.split(",")
    lang = "Java"
    print(f"Reading source examples from {source}.")
    print(f"Reading target examples from {target}.")

    with open(source, encoding="utf-8") as f1, open(target, encoding="utf-8") as f2:
        for line1, line2 in tqdm(zip(f1, f2)):
            line1 = line1.strip()
            line2 = line2.strip()
            examples.append(
                Example(
                    buggy_segment=line1,
                    target=line2, 
                    lang=lang, 
                    source_until="",
                    source_from="",
                    target_context="",
                )
            )

    print(f"Got {len(examples)} many examples.")
    return examples

def read_examples(path: str, without_python: bool = False) -> List[Example]:
    df = pd.read_csv(
        path,
        names=[
            "encoder_context_until",
            "buggy_segment",
            "encoder_context_from",
            "decoder_context",
            "fixed_segment",
            "author",
            "repo_name",
            "commit_hash",
            "language",
            "commit_date",
            "watch_count",
            "hash_id",
        ],
        dtype={
            "encoder_context_until": "string",
            "buggy_segment": "string",
            "encoder_context_from": "string",
            "decoder_context": "string",
            "fixed_segment": "string",
            "author": "string",
            "repo_name": "string",
            "commit_hash": "string",
            "language": "string",
            "commit_date": "string",
        },
    )
    df.fillna(" ", inplace=True)
    if without_python:
        df = df[df["language"] != "Python"]
    return [
        Example(
            row["encoder_context_until"],
            row["buggy_segment"],
            row["encoder_context_from"],
            row["fixed_segment"],
            row["decoder_context"],
            row["language"],
        )
        for _, row in df.iterrows()
    ]

def decode_safe(data: bytes):
    if data is None:
        return ""
    if encoding := cchardet.detect(data)["encoding"]:
        try:
            return data.decode(encoding=encoding)
        except UnicodeDecodeError as exception:
            print(exception)
            return ""
    return ""

def predict_bugs2fix(bugs2fix: str, csv: str, model_50_100: GraphCodeBERT, model_0_50: GraphCodeBERT):
    df = pd.read_csv(
        csv,
        dtype={
            "bid": "string",
            "pid": "string",
            "idx": "string",
            "abstract_path": "string",
            "buggy_path": "string",
            "bugs2fix_path": "string",
            "start_lineno": "int",
            "end_lineno": "int",
        },
    )

    for _, row in tqdm(df.iterrows(), total=len(df)):
        with open(os.path.join(bugs2fix, row["abstract_path"], "abstract"), "rb") as fstream:
            source = decode_safe(fstream.read())
            if source == "":
                continue
        example = Example(
            "",
            source,
            "",
            "",
            "",
            "Java",
        )
        tokenized = model_50_100.tokenize_examples([example], verbose=False)[0]
        if len(tokenized[0].code_tokens) < 56:
            prediction = model_0_50.predict_one_example(example)
        else:
            prediction = model_50_100.predict_one_example(example)
        with open(os.path.join(bugs2fix, row["abstract_path"], "prediction2"), "w") as fstream:
            fstream.write(prediction)

def train(model: Union[UnixCoder, GraphCodeBERT], train_file: str, valid_file: str, output_dir: str, learning_rate: float):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("read training dataset")
    train_samples = model.read_tokenized_dataset(train_file)
    print("read validation dataset")
    valid_samples = model.read_tokenized_dataset(valid_file)
    model.train_examples(
        train_samples, 
        output_dir, 
        validation=valid_samples, 
        batch_size=20,
        learning_rate=learning_rate,
    )

def learning(model: Union[UnixCoder, GraphCodeBERT], train_file: str):
    print("read training dataset")
    train_samples = model.read_tokenized_dataset(train_file)
    model.get_learning_rate(train_samples)

def tokenize(model: Union[UnixCoder, GraphCodeBERT], source: str, target: str, without_python: bool = False):
    examples = read_examples(source, without_python)
    print("creating tokenized dataset")
    model.create_tokenized_dataset(examples, target)

def tokenize_decoder(model: Union[UnixCoder, GraphCodeBERT], source: str, target: str, without_python: bool = False):
    examples = read_examples(source, without_python)
    print("creating tokenized dataset with encoder context as decoder context")
    model.create_tokenized_dataset_decoder(examples, target)

def test(model: Union[UnixCoder, GraphCodeBERT], test_file: str, output: str, load_model: str):
    print(f"Loading model from {load_model}.")
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(torch.load(load_model))

    print("read testing dataset")
    test_samples = model.read_tokenized_dataset(test_file)
    model.test_examples(test_samples, output)

def best_and_worst(model: Union[UnixCoder, GraphCodeBERT], test_file: str, output: str, load_model: str):
    print(f"Loading model from {load_model}.")
    model.load_state_dict(torch.load(load_model))

    print("read testing dataset")
    test_samples = model.read_tokenized_dataset(test_file)
    model.get_best_and_worst(test_samples, output)

def tokenize_bugs2fix(model: Union[UnixCoder, GraphCodeBERT], source: str, target: str):
    examples = read_examples_bugs2fix(source)
    print("creating tokenized dataset")
    model.create_tokenized_dataset(examples, target)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1]
    print(f"cmd is {cmd}")
    model_name = sys.argv[2]
    print(f"model is {model_name}")

    print("building model ...")
    if model_name == "graphcodebert":
        model = build_scratch_graphcodebert(preprocessor=delete_indentation, beam_size=10, max_window_size=3)
    elif model_name == "unixcoder":
        model = build_scratch_unixcoder(preprocessor=delete_indentation, beam_size=10, max_window_size=3)
    print("Done")

    if cmd == "try":
        try_model(model)
    elif cmd == "train":
        train_file = sys.argv[3]
        valid_file = sys.argv[4]
        output_dir = sys.argv[5]
        learning_rate = float(sys.argv[6])

        if len(sys.argv) >= 8:
            limitation = sys.argv[7]
        else:
            limitation = ""
        print(f"Limitation is {limitation}")
        
        if limitation == "nowindow":
            print("Disabling winowization")
            model.windowizer = no_windowization
        elif limitation == "nocontext":
            print("Disabling context")
            model.preprocessor = no_context 
        elif limitation == "noboth":
            print("Disabling winowization")
            model.windowizer = no_windowization
            print("Disabling context")
            model.preprocessor = no_context 
        else:
            print(f"limitation {limitation} does not exist.")


        print(f"training file is {train_file}")
        print(f"validation file is {valid_file}")
        print(f"output directory is {output_dir}")
        print(f"Learning rate is {learning_rate}")
        train(model, train_file, valid_file, output_dir, learning_rate)
    elif cmd == "tokenize":
        source = sys.argv[3]
        target = sys.argv[4]
        if len(sys.argv) == 6 and sys.argv[5] == "no_python":
            without_python = True
            print("Ignoring Python samples.")
        else:
            without_python = False
        if len(sys.argv) == 6 and sys.argv[5] == "no_cutter":
            model.precutter = no_cutter
            print("Not using any precutter")
        print(f"Tokenizing {source} and saving it in {target}")
        tokenize(model, source, target, without_python)
    elif cmd == "tokenize_decoder":
        source = sys.argv[3]
        target = sys.argv[4]
        if len(sys.argv) == 6 and sys.argv[5] == "no_python":
            without_python = True
            print("Ignoring Python samples.")
        else:
            without_python = False
        if len(sys.argv) == 6 and sys.argv[5] == "no_cutter":
            model.precutter = no_cutter
            print("Not using any precutter")
        print(f"Tokenizing {source} and saving it in {target}")
        tokenize_decoder(model, source, target, without_python)
    elif cmd == "tokenize_bugs2fix":
        source = sys.argv[3]
        target = sys.argv[4]
        print(f"Tokenizing {source} and saving it in {target}")
        tokenize_bugs2fix(model, source, target)   
    elif cmd == "learning":
        train_file = sys.argv[3]
        print(f"training file is {train_file}")
        learning(model, train_file)
    elif cmd == "test":
        test_file = sys.argv[3]
        output_dir = sys.argv[4]
        load_model = sys.argv[5]

        if len(sys.argv) >= 7:
            limitation = sys.argv[6]
        else:
            limitation = ""
        print(f"Limitation is {limitation}")
        
        if limitation == "nowindow":
            print("Disabling winowization")
            model.windowizer = no_windowization
        elif limitation == "nocontext":
            print("Disabling context")
            model.preprocessor = no_context 
        elif limitation == "noboth":
            print("Disabling winowization")
            model.windowizer = no_windowization
            print("Disabling context")
            model.preprocessor = no_context 
        else:
            print(f"limitation {limitation} does not exist.")
        
        print(f"testing file is {test_file}")
        print(f"output directory is {output_dir}")
        print(f"load model is {load_model}")

        test(model, test_file, output_dir, load_model)
    elif cmd == "defects4j":
        source_path = sys.argv[3] 
        target_path = sys.argv[4] 
        load_model = sys.argv[5]
        print(f"source path is {source_path}")
        print(f"target path is {target_path}")
        print(f"load model is {load_model}")
        model.load_state_dict(torch.load(load_model))
        predict_defects4j(source_path, target_path, model)
    elif cmd == "best_worst":
        source_path = sys.argv[3]
        output_dir = sys.argv[4]
        load_model = sys.argv[5]
        print(f"source path is {source_path}")
        print(f"output dir is {output_dir}")
        print(f"load model is {load_model}")
        best_and_worst(model, source_path, output_dir, load_model)
    elif cmd == "weights":
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        num_params = count_parameters(model)
        print(f"Number of trainable parameters in the model: {num_params}")
    elif cmd == "read":
        source_path = sys.argv[3]
        print(f"source path is {source_path}")
        model.read_tokenized_dataset(source_path)
    elif cmd == "bugs2fix":
        bugs2fix_path = sys.argv[3]
        csv_path = sys.argv[4]
        load1 = sys.argv[5]
        load2 = sys.argv[6]
        print(f"Bugs2fix path is {bugs2fix_path}.")
        print(f"csv path is {csv_path}.")
        print(f"Model for 50-100 tokens {load1}.")
        print(f"Model for 0-50 tokens {load2}.")

        if model_name == "graphcodebert":
            model_0_50 = build_scratch_graphcodebert(preprocessor=delete_indentation, beam_size=10, max_window_size=3)
        elif model_name == "unixcoder":
            model_0_50 = build_scratch_unixcoder(preprocessor=delete_indentation, beam_size=10, max_window_size=3)

        model_50_100 = model
        model_50_100.load_state_dict(torch.load(load1))
        model_0_50.load_state_dict(torch.load(load2))
        predict_bugs2fix(bugs2fix_path, csv_path, model_50_100, model_0_50) 

            

