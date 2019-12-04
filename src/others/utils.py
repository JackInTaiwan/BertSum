import os
import re
import shutil
import time

from rouge import Rouge

from src.others.logging import logger

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}



def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        logger.info(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding="utf-8")]
    references = [line.strip() for line in open(ref, encoding="utf-8")]

    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])

        candidates_, references_ = [], []
        for cand, ref in zip(candidates, references):
            if len(cand) * len(ref) != 0:
                cand = " ".join(cand.split("<q>"))
                ref = " ".join(ref.split("<q>"))
                candidates_.append(cand)
                references_.append(ref)

        # calculate rouge score
        logger.info("| Calculating rouge score on {} candidate-reference pairs ...".format(len(candidates_)))

        rouge = Rouge()
        rouge_result_dict = rouge.get_scores(candidates_, references_, avg=True)

    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    return rouge_result_dict


def rouge_results_to_str(results_dict):
    return "| ROUGE-F1(1/2/l): {:.4f}/{:.4f}/{:.4f}\n| ROUGE-R(1/2/l): {:.4f}/{:.4f}/{:.4f}\n".format(
        results_dict["rouge-1"]["f"],
        results_dict["rouge-2"]["f"],
        results_dict["rouge-l"]["f"],
        results_dict["rouge-1"]["r"],
        results_dict["rouge-2"]["r"],
        results_dict["rouge-l"]["r"]
    )
