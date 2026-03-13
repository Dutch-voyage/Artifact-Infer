from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify import parse, verify
from typing import Callable, Any
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from collections import Counter


def evaluate(answer, ground_truth):
    
    # verify_func = math_metric(
    #     gold_extraction_target=(LatexExtractionConfig(),),
    #     pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    # )
    preds = []
    
    if isinstance(answer, list):
        # choose the concensus
        parse_result = [parse(pred, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]) for pred in answer]
        pred_list = ["\\boxed{" + parse_result[i][1] + "}" if len(parse_result[i]) > 1 else None for i in range(len(parse_result))]
        preds = Counter(pred_list).most_common(1)[0][0]
    else:
        parse_result = parse(answer, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
        preds = parse_result[1] if len(parse_result) > 1 else None
    
    if preds is None:
        return 0.0, None
    
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    preds = parse(preds, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
    gold = parse(ground_truth_boxed, extraction_config=[LatexExtractionConfig()])
    if verify(gold, preds, 6):
        ret_score = 1.0
    # try:
    #     ret_score, _ = verify_func([ground_truth_boxed], [answer])
    # except Exception:
    #     pass
    # except TimeoutException:
    #     pass

    return ret_score, preds
