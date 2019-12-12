import os
import torch

from argparse import ArgumentParser



def dump_eyeball_file(candidate_fp, oracle_fp, bert_data_fp, eyeball_number):
    bert_data = torch.load(bert_data_fp)

    output_fp = os.path.join(os.path.dirname(candidate_fp), "_eyeball_{}.txt".format(eyeball_number))
    
    with open(candidate_fp) as c, open(oracle_fp) as ora, open(output_fp, "w") as out:
        for i, data in enumerate(bert_data[:eyeball_number]):
            src_txt, tgt_txt = data["src_txt"], data["tgt_txt"]
            candidates = c.readline().strip()
            oracle = ora.readline().strip()
            out.writelines([
                "[[[{}]]]\n".format(i),
                "[Source Text]\n",
                "{}\n\n".format("<q>".join(src_txt)),
                "[BertSum Summary]\n",
                "{}\n\n".format(candidates),
                "[Oracle Summary]\n",
                "{}\n\n".format(oracle),
                "[Golden Summary]\n",
                "{}\n\n".format(tgt_txt),
                "\n\n\n"
            ])



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--candidate_fp", type=str, required=True)
    parser.add_argument("-o", "--oracle_fp", type=str, required=True)
    parser.add_argument("-b", "--bert_data_fp", type=str, required=True)
    parser.add_argument("-n", "--eyeball_number", type=int, default=200)

    args = parser.parse_args()
    
    dump_eyeball_file(args.candidate_fp, args.oracle_fp, args.bert_data_fp, args.eyeball_number)