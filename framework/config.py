from framework.processes import ProcessStructure, Task, Resource, Gateway


import random
import string
from collections import defaultdict

# # Placeholder classes for ProcessStructure, Task, Gateway, Resource
# class Task:
#     def __init__(self, name, resources=None, next_tasks=None):
#         self.name = name
#         self.resources = resources or []
#         self.next_tasks = next_tasks or []

# class Gateway:
#     def __init__(self, name, gateway_type, next_tasks, conditions):
#         self.name = name
#         self.gateway_type = gateway_type
#         self.next_tasks = next_tasks
#         self.conditions = conditions

# class Resource:
#     def __init__(self, name, exec_distribution):
#         self.name = name
#         self.exec_distribution = exec_distribution

# class ProcessStructure:
#     def __init__(self, name, arrival_distribution, data_options, tasks):
#         self.name = name
#         self.arrival_distribution = arrival_distribution
#         self.data_options = data_options
#         self.tasks = tasks
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Callable, Tuple


def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                   
set_seed(42) 





def process_function(l:float, scenario:str):
    low=0.5
    high=2.0
    if scenario=='scenario_1_A': 
        processes = [
            ProcessStructure(
                name="p_1",
                arrival_distribution=l,
                data_options={
                    "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17":0.1},
                    "priority2": {"p1_END": 0.3, "a_18": 0.7},
                    "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2":0.1, "a_1":0.2},
                    "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21":0.1, "a_22":0.2}, 
                    "request_type": {"a_12": 0.4, "a_13": 0.6},
                    "loop1": {"a_3": 0.05, "p1_END": 0.95},
                    "loop2": {"a_16": 0.05, "a_15": 0.95},
                    "loop3": {"a_10": 0.05, "a_12": 0.95},
                    "loop4": {"a_8": 0.05, "a_14": 0.95},
                    "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
                    "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
                    "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
                    "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
                    "quality_flag": {"q_1": 0.7, "q_2": 0.3},
                    "report_needed": {"r_1": 0.4, "r_skip": 0.6}
                },
                tasks={
                    "START": Task(name="a_start", next_tasks=["a_1"]),
                    "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),
                    "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
                                    next_tasks=["a_2", "a_3", "a_4", "a_17"],
                                    conditions=["priority"]),
                    "a_2": Task(name="a_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
                    "a_3": Task(name="a_3", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
                    "a_3_1": Task(name="a_3_1", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),
                    "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
                                    next_tasks=["a_12", "a_6"],
                                    conditions=["or_sub_1"]),
                    "a_4": Task(name="a_4", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

                    "a_5": Task(name="a_5", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
                    "a_6": Task(name="a_6", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

                    "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
                    "a6_diag": Task(name="a6_diag", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_log": Task(name="a6_log", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_notify": Task(name="a6_notify", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),


                    "a_7": Task(name="a_7", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

                    "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
                                    next_tasks=["a_8", "a_9", "a_10", "r_2","a_1"],
                                    conditions=["department"]),

                    "a_8": Task(name="a_8", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
                    "a_9": Task(name="a_9", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
                    "a_10": Task(name="a_10", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),
                    
                    "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
                    "a10_qc": Task(name="a10_qc", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "a10_trace": Task(name="a10_trace", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
                    "a10_email": Task(name="a10_email", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),





                    "a_11": Task(name="a_11", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

                    "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
                                    next_tasks=["a_12", "a_13"],
                                    conditions=["request_type"]),
                    "a_12": Task(name="a_12", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
                    "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
                                    next_tasks=["a_12_1", "a_12_2"],
                                    conditions=["or_sub_3"]),

                    "a_12_1": Task(name="a_12_1", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
                    "a_12_2": Task(name="a_12_2", resources=[Resource("int22", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),




                    "a_13": Task(name="a_13", resources=[Resource("int23", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
                    "a_131": Task(name="a_131", resources=[Resource("int24", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
                    "a_132": Task(name="a_132", resources=[Resource("int25", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
                    "a_14": Task(name="a_14", resources=[Resource("int26", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

                    # New Gateway: Quality Check Branch
                    "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
                        next_tasks=["q_1", "q_2"],
                        conditions=["quality_flag"]),

                    # New Gateway: Escalation Decision
                    "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
                        next_tasks=["e_1", "e_2", "e_3"],
                        conditions=["escalation_level"]),

                    # New Gateway: Reporting Needed
                    "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
                        next_tasks=["r_1", "r_skip"],
                        conditions=["report_needed"]),

                    # New Gateway: Final Loop Check (re-entry)
                    "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
                        next_tasks=["a_3", "p1_END"],
                        conditions=["loop1"]),

                    # New Tasks
                    "q_1": Task(name="q_1", resources=[Resource("int27", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
                    "q_2": Task(name="q_2", resources=[Resource("int28", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

                    "e_1": Task(name="e_1", resources=[Resource("int29", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
                    "e_2": Task(name="e_2", resources=[Resource("int30", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
                    "e_3": Task(name="e_3", resources=[Resource("int31", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

                    "r_1": Task(name="r_1", resources=[Resource("int32", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
                    "r_skip": Task(name="r_skip", resources=[Resource("int33", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

                    "r_2": Task(name="r_2", resources=[Resource("int34", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

                    "a_24": Task(name="a_24", resources=[Resource("int35", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
                    "a_25": Task(name="a_25", resources=[Resource("int36", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
                    "a_26": Task(name="a_26", resources=[Resource("int37", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

                    "a_27": Task(name="a_27", resources=[Resource("int38", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
                    "a_28": Task(name="a_28", resources=[Resource("int39", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
                    "a_29": Task(name="a_29", resources=[Resource("int40", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),
                    

                    "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
                                    next_tasks=["a_16", "a_15"],
                                    conditions=["loop2"]),
                    "a_15": Task(name="a_15", resources=[Resource("int41", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
                    "a_16": Task(name="a_16", resources=[Resource("int42", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),  

                    "a_17": Task(name="a_17", resources=[Resource("int43", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),
                    "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
                                    next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
                                    conditions=["or_sub_2"]),
                    "a_18": Task(name="a_18", resources=[Resource("int44", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

                    "a_19": Task(name="a_19", resources=[Resource("int45", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
                    "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
                                        next_tasks=["a_19_1", "a_19_2", "p1_END"],
                                        conditions=["post_processing"]),

                    "a_19_1": Task(name="a_19_1", resources=[Resource("int46", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
                    "a_19_2": Task(name="a_19_2", resources=[Resource("int47", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),
                    "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
                                        next_tasks=["a_10", "a_12"],
                                        conditions=["loop3"]),

                    "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
                                        next_tasks=["p1_END", "a_18"],
                                        conditions=["priority2"]),

                    "a_21": Task(name="a_21", resources=[Resource("int48", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
                    "a_22": Task(name="a_22", resources=[Resource("int49", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
                    "a_23": Task(name="a_23", resources=[Resource("int50", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),
                    "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND", next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),
                    "a23_notify": Task(name="a23_notify", resources=[Resource("int51", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_audit": Task(name="a23_audit", resources=[Resource("int52", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_escalate": Task(name="a23_escalate", resources=[Resource("int53", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND", merge_from=["a23_notify", "a23_audit", "a23_escalate"], next_tasks=["p1_END"]),

                    "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
                                    next_tasks=["a_8", "a_14"],
                                    conditions=["loop4"]),
                    "p1_END": Task(name="p1_END", next_tasks=[])
                }
            )

            ,
            ProcessStructure(
                name="p_2",
                arrival_distribution=l,
                data_options={
                    "case_type": {"b_2": 0.5, "b_3": 0.5},
                    "severity": {"b_6": 0.3, "b_7": 0.4, "b_8": 0.3},
                    "severity2": {"b_13": 0.2, "b_14": 0.5, "p2_END": 0.3},
                    "needs_review": {"b_11": 0.4, "b_12": 0.6},
                    "review": {"b_16": 0.2, "b_17": 0.2, "b_18":0.3, "p2_END":0.3},
                },
                tasks={
                    "START": Task(name="b_start", next_tasks=["b_1"]),
                    "b_1": Task(name="b_1", resources=[Resource("i1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_b1"]),
                    "XOR_b1": Gateway(name="XOR_b1", gateway_type="XOR",
                                    next_tasks=["b_2", "b_3"],
                                    conditions=["case_type"]),
                    "b_2": Task(name="b_2", resources=[Resource("i2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_4"]),
                    "b_3": Task(name="b_3", resources=[Resource("i3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_4"]),
                    "b_4": Task(name="b_4", resources=[Resource("i4", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["b_5"]),

                    "b_5": Task(name="b_5", resources=[Resource("i5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_b2"]),
                    "XOR_b2": Gateway(name="XOR_b2", gateway_type="XOR",
                                    next_tasks=["b_6", "b_7", "b_8"],
                                    conditions=["severity"]),
                    "b_6": Task(name="b_6", resources=[Resource("i6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_10"]),
                    "b_7": Task(name="b_7", resources=[Resource("i7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_10"]),
                    "b_8": Task(name="b_8", resources=[Resource("i8", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["b_10"]),
                    "b_10": Task(name="b_10", resources=[Resource("i9", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_b3"]),

                    "XOR_b3": Gateway(name="XOR_b3", gateway_type="XOR",
                                    next_tasks=["b_11", "b_12"],
                                    conditions=["needs_review"]),
                    "b_11": Task(name="b_11", resources=[Resource("i10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["b_9"]),  # loop
                    "b_9": Task(name="b_9", resources=[Resource("i11", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_b4"]),

                    "b_12": Task(name="b_12", resources=[Resource("i12", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_b4"]),

                    "XOR_b4": Gateway(name="XOR_b4", gateway_type="XOR",
                                    next_tasks=["b_13", "b_14", "p2_END"],
                                    conditions=["severity2"]),
                    "b_13": Task(name="b_13", resources=[Resource("i13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_15"]),
                    "b_14": Task(name="b_14", resources=[Resource("i14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["b_15"]),
                    "b_15": Task(name="b_15", resources=[Resource("i15", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_b5"]),

                    "XOR_b5": Gateway(name="XOR_b5", gateway_type="XOR",
                                    next_tasks=["b_16", "b_17", "b_18", "p2_END"],
                                    conditions=["review"]),
                    "b_16": Task(name="b_16", resources=[Resource("i16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p2_END"]),
                    "b_17": Task(name="b_17", resources=[Resource("i17", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["p2_END"]),
                    "b_18": Task(name="b_18", resources=[Resource("i18", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["p2_END"]),
                    "p2_END": Task(name="p2_END", next_tasks=[])
                }
            )
        ]

        return processes

    

    if scenario=='scenario_1_B': # Sharing all resources
        processes = [
            ProcessStructure(
                name="p_1",
                arrival_distribution=l,
                data_options={
                    "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17":0.1},
                    "priority2": {"p1_END": 0.3, "a_18": 0.7},
                    "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2":0.1, "a_1":0.2},
                    "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21":0.1, "a_22":0.2}, 
                    "request_type": {"a_12": 0.4, "a_13": 0.6},
                    "loop1": {"a_3": 0.05, "p1_END": 0.95},
                    "loop2": {"a_16": 0.05, "a_15": 0.95},
                    "loop3": {"a_10": 0.05, "a_12": 0.95},
                    "loop4": {"a_8": 0.05, "a_14": 0.95},
                    "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
                    "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
                    "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
                    "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
                    "quality_flag": {"q_1": 0.7, "q_2": 0.3},
                    "report_needed": {"r_1": 0.4, "r_skip": 0.6}
                },
                tasks={
                    "START": Task(name="a_start", next_tasks=["a_1"]),
                    "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),
                    "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
                                    next_tasks=["a_2", "a_3", "a_4", "a_17"],
                                    conditions=["priority"]),
                    "a_2": Task(name="a_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
                    "a_3": Task(name="a_3", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
                    "a_3_1": Task(name="a_3_1", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),
                    "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
                                    next_tasks=["a_12", "a_6"],
                                    conditions=["or_sub_1"]),
                    "a_4": Task(name="a_4", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

                    "a_5": Task(name="a_5", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
                    "a_6": Task(name="a_6", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

                    "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
                    "a6_diag": Task(name="a6_diag", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_log": Task(name="a6_log", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_notify": Task(name="a6_notify", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),


                    "a_7": Task(name="a_7", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

                    "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
                                    next_tasks=["a_8", "a_9", "a_10", "r_2","a_1"],
                                    conditions=["department"]),

                    "a_8": Task(name="a_8", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
                    "a_9": Task(name="a_9", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
                    "a_10": Task(name="a_10", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),
                    
                    "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
                    "a10_qc": Task(name="a10_qc", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "a10_trace": Task(name="a10_trace", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
                    "a10_email": Task(name="a10_email", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),





                    "a_11": Task(name="a_11", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

                    "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
                                    next_tasks=["a_12", "a_13"],
                                    conditions=["request_type"]),
                    "a_12": Task(name="a_12", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
                    "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
                                    next_tasks=["a_12_1", "a_12_2"],
                                    conditions=["or_sub_3"]),

                    "a_12_1": Task(name="a_12_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
                    "a_12_2": Task(name="a_12_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),




                    "a_13": Task(name="a_13", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
                    "a_131": Task(name="a_131", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
                    "a_132": Task(name="a_132", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
                    "a_14": Task(name="a_14", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

                    # New Gateway: Quality Check Branch
                    "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
                        next_tasks=["q_1", "q_2"],
                        conditions=["quality_flag"]),

                    # New Gateway: Escalation Decision
                    "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
                        next_tasks=["e_1", "e_2", "e_3"],
                        conditions=["escalation_level"]),

                    # New Gateway: Reporting Needed
                    "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
                        next_tasks=["r_1", "r_skip"],
                        conditions=["report_needed"]),

                    # New Gateway: Final Loop Check (re-entry)
                    "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
                        next_tasks=["a_3", "p1_END"],
                        conditions=["loop1"]),

                    # New Tasks
                    "q_1": Task(name="q_1", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
                    "q_2": Task(name="q_2", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

                    "e_1": Task(name="e_1", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
                    "e_2": Task(name="e_2", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
                    "e_3": Task(name="e_3", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

                    "r_1": Task(name="r_1", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
                    "r_skip": Task(name="r_skip", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

                    "r_2": Task(name="r_2", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

                    "a_24": Task(name="a_24", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
                    "a_25": Task(name="a_25", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
                    "a_26": Task(name="a_26", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

                    "a_27": Task(name="a_27", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
                    "a_28": Task(name="a_28", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
                    "a_29": Task(name="a_29", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),
                    

                    "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
                                    next_tasks=["a_16", "a_15"],
                                    conditions=["loop2"]),
                    "a_15": Task(name="a_15", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
                    "a_16": Task(name="a_16", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),  

                    "a_17": Task(name="a_17", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),
                    "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
                                    next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
                                    conditions=["or_sub_2"]),
                    "a_18": Task(name="a_18", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

                    "a_19": Task(name="a_19", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
                    "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
                                        next_tasks=["a_19_1", "a_19_2", "p1_END"],
                                        conditions=["post_processing"]),

                    "a_19_1": Task(name="a_19_1", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
                    "a_19_2": Task(name="a_19_2", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),
                    "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
                                        next_tasks=["a_10", "a_12"],
                                        conditions=["loop3"]),

                    "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
                                        next_tasks=["p1_END", "a_18"],
                                        conditions=["priority2"]),

                    "a_21": Task(name="a_21", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
                    "a_22": Task(name="a_22", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
                    "a_23": Task(name="a_23", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),
                    "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND", next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),
                    "a23_notify": Task(name="a23_notify", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_audit": Task(name="a23_audit", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_escalate": Task(name="a23_escalate", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND", merge_from=["a23_notify", "a23_audit", "a23_escalate"], next_tasks=["p1_END"]),

                    "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
                                    next_tasks=["a_8", "a_14"],
                                    conditions=["loop4"]),
                    "p1_END": Task(name="p1_END", next_tasks=[])
                }
            )

            ,
            ProcessStructure(
                name="p_2",
                arrival_distribution=l,
                data_options={
                    "case_type": {"b_2": 0.5, "b_3": 0.5},
                    "severity": {"b_6": 0.3, "b_7": 0.4, "b_8": 0.3},
                    "severity2": {"b_13": 0.2, "b_14": 0.5, "p2_END": 0.3},
                    "needs_review": {"b_11": 0.4, "b_12": 0.6},
                    "review": {"b_16": 0.2, "b_17": 0.2, "b_18":0.3, "p2_END":0.3},
                },
                tasks={
                    "START": Task(name="b_start", next_tasks=["b_1"]),
                    "b_1": Task(name="b_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_b1"]),
                    "XOR_b1": Gateway(name="XOR_b1", gateway_type="XOR",
                                    next_tasks=["b_2", "b_3"],
                                    conditions=["case_type"]),
                    "b_2": Task(name="b_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_4"]),
                    "b_3": Task(name="b_3", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_4"]),
                    "b_4": Task(name="b_4", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["b_5"]),

                    "b_5": Task(name="b_5", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_b2"]),
                    "XOR_b2": Gateway(name="XOR_b2", gateway_type="XOR",
                                    next_tasks=["b_6", "b_7", "b_8"],
                                    conditions=["severity"]),
                    "b_6": Task(name="b_6", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_10"]),
                    "b_7": Task(name="b_7", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_10"]),
                    "b_8": Task(name="b_8", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["b_10"]),
                    "b_10": Task(name="b_10", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_b3"]),

                    "XOR_b3": Gateway(name="XOR_b3", gateway_type="XOR",
                                    next_tasks=["b_11", "b_12"],
                                    conditions=["needs_review"]),
                    "b_11": Task(name="b_11", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["b_9"]),  # loop
                    "b_9": Task(name="b_9", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_c_4"]),

                    "b_12": Task(name="b_12", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_c_4"]),

                    "XOR_c_4": Gateway(name="XOR_c_4", gateway_type="XOR",
                                    next_tasks=["b_13", "b_14", "p2_END"],
                                    conditions=["severity2"]),
                    "b_13": Task(name="b_13", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["b_15"]),
                    "b_14": Task(name="b_14", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["b_15"]),
                    "b_15": Task(name="b_15", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_b5"]),

                    "XOR_b5": Gateway(name="XOR_b5", gateway_type="XOR",
                                    next_tasks=["b_16", "b_17", "b_18", "p2_END"],
                                    conditions=["review"]),
                    "b_16": Task(name="b_16", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p2_END"]),
                    "b_17": Task(name="b_17", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["p2_END"]),
                    "b_18": Task(name="b_18", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["p2_END"]),
                    "p2_END": Task(name="p2_END", next_tasks=[])
                }
            )
        ]

        return processes

    if scenario=='scenario_1_B_75_unique': 
        processes = [
            ProcessStructure(
                name="p_1",
                arrival_distribution=l,
                data_options={
                    "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17":0.1},
                    "priority2": {"p1_END": 0.3, "a_18": 0.7},
                    "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2":0.1, "a_1":0.2},
                    "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21":0.1, "a_22":0.2}, 
                    "request_type": {"a_12": 0.4, "a_13": 0.6},
                    "loop1": {"a_3": 0.05, "p1_END": 0.95},
                    "loop2": {"a_16": 0.05, "a_15": 0.95},
                    "loop3": {"a_10": 0.05, "a_12": 0.95},
                    "loop4": {"a_8": 0.05, "a_14": 0.95},
                    "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
                    "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
                    "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
                    "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
                    "quality_flag": {"q_1": 0.7, "q_2": 0.3},
                    "report_needed": {"r_1": 0.4, "r_skip": 0.6}
                },
                tasks={
                    "START": Task(name="a_start", next_tasks=["a_1"]),
                    "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),
                    "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
                                    next_tasks=["a_2", "a_3", "a_4", "a_17"],
                                    conditions=["priority"]),
                    "a_2": Task(name="a_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
                    "a_3": Task(name="a_3", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
                    "a_3_1": Task(name="a_3_1", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),
                    "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
                                    next_tasks=["a_12", "a_6"],
                                    conditions=["or_sub_1"]),
                    "a_4": Task(name="a_4", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

                    "a_5": Task(name="a_5", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
                    "a_6": Task(name="a_6", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

                    "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
                    "a6_diag": Task(name="a6_diag", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_log": Task(name="a6_log", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_notify": Task(name="a6_notify", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),


                    "a_7": Task(name="a_7", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

                    "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
                                    next_tasks=["a_8", "a_9", "a_10", "r_2","a_1"],
                                    conditions=["department"]),

                    "a_8": Task(name="a_8", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
                    "a_9": Task(name="a_9", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
                    "a_10": Task(name="a_10", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),
                    
                    "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
                    "a10_qc": Task(name="a10_qc", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "a10_trace": Task(name="a10_trace", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
                    "a10_email": Task(name="a10_email", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),





                    "a_11": Task(name="a_11", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

                    "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
                                    next_tasks=["a_12", "a_13"],
                                    conditions=["request_type"]),
                    "a_12": Task(name="a_12", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
                    "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
                                    next_tasks=["a_12_1", "a_12_2"],
                                    conditions=["or_sub_3"]),

                    "a_12_1": Task(name="a_12_1", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
                    "a_12_2": Task(name="a_12_2", resources=[Resource("int22", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),




                    "a_13": Task(name="a_13", resources=[Resource("int23", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
                    "a_131": Task(name="a_131", resources=[Resource("int24", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
                    "a_132": Task(name="a_132", resources=[Resource("int25", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
                    "a_14": Task(name="a_14", resources=[Resource("int26", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

                    # New Gateway: Quality Check Branch
                    "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
                        next_tasks=["q_1", "q_2"],
                        conditions=["quality_flag"]),

                    # New Gateway: Escalation Decision
                    "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
                        next_tasks=["e_1", "e_2", "e_3"],
                        conditions=["escalation_level"]),

                    # New Gateway: Reporting Needed
                    "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
                        next_tasks=["r_1", "r_skip"],
                        conditions=["report_needed"]),

                    # New Gateway: Final Loop Check (re-entry)
                    "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
                        next_tasks=["a_3", "p1_END"],
                        conditions=["loop1"]),

                    # New Tasks
                    "q_1": Task(name="q_1", resources=[Resource("int27", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
                    "q_2": Task(name="q_2", resources=[Resource("int28", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

                    "e_1": Task(name="e_1", resources=[Resource("int29", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
                    "e_2": Task(name="e_2", resources=[Resource("int30", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
                    "e_3": Task(name="e_3", resources=[Resource("int31", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

                    "r_1": Task(name="r_1", resources=[Resource("int32", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
                    "r_skip": Task(name="r_skip", resources=[Resource("int33", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

                    "r_2": Task(name="r_2", resources=[Resource("int34", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

                    "a_24": Task(name="a_24", resources=[Resource("int35", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
                    "a_25": Task(name="a_25", resources=[Resource("int36", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
                    "a_26": Task(name="a_26", resources=[Resource("int37", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

                    "a_27": Task(name="a_27", resources=[Resource("int38", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
                    "a_28": Task(name="a_28", resources=[Resource("int39", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
                    "a_29": Task(name="a_29", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),
                    

                    "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
                                    next_tasks=["a_16", "a_15"],
                                    conditions=["loop2"]),
                    "a_15": Task(name="a_15", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
                    "a_16": Task(name="a_16", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),  

                    "a_17": Task(name="a_17", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),
                    "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
                                    next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
                                    conditions=["or_sub_2"]),
                    "a_18": Task(name="a_18", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

                    "a_19": Task(name="a_19", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
                    "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
                                        next_tasks=["a_19_1", "a_19_2", "p1_END"],
                                        conditions=["post_processing"]),

                    "a_19_1": Task(name="a_19_1", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
                    "a_19_2": Task(name="a_19_2", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),
                    "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
                                        next_tasks=["a_10", "a_12"],
                                        conditions=["loop3"]),

                    "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
                                        next_tasks=["p1_END", "a_18"],
                                        conditions=["priority2"]),

                    "a_21": Task(name="a_21", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
                    "a_22": Task(name="a_22", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
                    "a_23": Task(name="a_23", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),
                    "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND", next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),
                    "a23_notify": Task(name="a23_notify", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_audit": Task(name="a23_audit", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_escalate": Task(name="a23_escalate", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND", merge_from=["a23_notify", "a23_audit", "a23_escalate"], next_tasks=["p1_END"]),

                    "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
                                    next_tasks=["a_8", "a_14"],
                                    conditions=["loop4"]),
                    "p1_END": Task(name="p1_END", next_tasks=[])
                }
            )]
        return processes


#### Contention resources works on multiple tasks
    if scenario=='scenario_1_B_20_unique': 
        low=0.5
        high=2.0
        processes = [
            ProcessStructure(
                name="p_1",
                arrival_distribution=l,
                data_options={
                    "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17":0.1},
                    "priority2": {"p1_END": 0.3, "a_18": 0.7},
                    "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2":0.1, "a_1":0.2},
                    "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21":0.1, "a_22":0.2}, 
                    "request_type": {"a_12": 0.4, "a_13": 0.6},
                    "loop1": {"a_3": 0.05, "p1_END": 0.95},
                    "loop2": {"a_16": 0.05, "a_15": 0.95},
                    "loop3": {"a_10": 0.05, "a_12": 0.95},
                    "loop4": {"a_8": 0.05, "a_14": 0.95},
                    "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
                    "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
                    "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
                    "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
                    "quality_flag": {"q_1": 0.7, "q_2": 0.3},
                    "report_needed": {"r_1": 0.4, "r_skip": 0.6}
                },
                tasks={
                    "START": Task(name="a_start", next_tasks=["a_1"]),
                    "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),
                    "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
                                    next_tasks=["a_2", "a_3", "a_4", "a_17"],
                                    conditions=["priority"]),
                    "a_2": Task(name="a_2", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
                    "a_3": Task(name="a_3", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
                    "a_3_1": Task(name="a_3_1", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),
                    "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
                                    next_tasks=["a_12", "a_6"],
                                    conditions=["or_sub_1"]),
                    "a_4": Task(name="a_4", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

                    "a_5": Task(name="a_5", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
                    "a_6": Task(name="a_6", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

                    "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
                    "a6_diag": Task(name="a6_diag", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_log": Task(name="a6_log", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_notify": Task(name="a6_notify", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),


                    "a_7": Task(name="a_7", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

                    "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
                                    next_tasks=["a_8", "a_9", "a_10", "r_2","a_1"],
                                    conditions=["department"]),

                    "a_8": Task(name="a_8", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
                    "a_9": Task(name="a_9", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
                    "a_10": Task(name="a_10", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),
                    
                    "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
                    "a10_qc": Task(name="a10_qc", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "a10_trace": Task(name="a10_trace", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
                    "a10_email": Task(name="a10_email", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),





                    "a_11": Task(name="a_11", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

                    "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
                                    next_tasks=["a_12", "a_13"],
                                    conditions=["request_type"]),
                    "a_12": Task(name="a_12", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
                    "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
                                    next_tasks=["a_12_1", "a_12_2"],
                                    conditions=["or_sub_3"]),

                    "a_12_1": Task(name="a_12_1", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
                    "a_12_2": Task(name="a_12_2", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),




                    "a_13": Task(name="a_13", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
                    "a_131": Task(name="a_131", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
                    "a_132": Task(name="a_132", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
                    "a_14": Task(name="a_14", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

                    # New Gateway: Quality Check Branch
                    "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
                        next_tasks=["q_1", "q_2"],
                        conditions=["quality_flag"]),

                    # New Gateway: Escalation Decision
                    "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
                        next_tasks=["e_1", "e_2", "e_3"],
                        conditions=["escalation_level"]),

                    # New Gateway: Reporting Needed
                    "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
                        next_tasks=["r_1", "r_skip"],
                        conditions=["report_needed"]),

                    # New Gateway: Final Loop Check (re-entry)
                    "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
                        next_tasks=["a_3", "p1_END"],
                        conditions=["loop1"]),

                    # New Tasks
                    "q_1": Task(name="q_1", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
                    "q_2": Task(name="q_2", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

                    "e_1": Task(name="e_1", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
                    "e_2": Task(name="e_2", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
                    "e_3": Task(name="e_3", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

                    "r_1": Task(name="r_1", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
                    "r_skip": Task(name="r_skip", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

                    "r_2": Task(name="r_2", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

                    "a_24": Task(name="a_24", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
                    "a_25": Task(name="a_25", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
                    "a_26": Task(name="a_26", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

                    "a_27": Task(name="a_27", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
                    "a_28": Task(name="a_28", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
                    "a_29": Task(name="a_29", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),
                    

                    "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
                                    next_tasks=["a_16", "a_15"],
                                    conditions=["loop2"]),
                    "a_15": Task(name="a_15", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
                    "a_16": Task(name="a_16", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),  

                    "a_17": Task(name="a_17", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),
                    "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
                                    next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
                                    conditions=["or_sub_2"]),
                    "a_18": Task(name="a_18", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

                    "a_19": Task(name="a_19", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
                    "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
                                        next_tasks=["a_19_1", "a_19_2", "p1_END"],
                                        conditions=["post_processing"]),

                    "a_19_1": Task(name="a_19_1", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
                    "a_19_2": Task(name="a_19_2", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),
                    "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
                                        next_tasks=["a_10", "a_12"],
                                        conditions=["loop3"]),

                    "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
                                        next_tasks=["p1_END", "a_18"],
                                        conditions=["priority2"]),

                    "a_21": Task(name="a_21", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
                    "a_22": Task(name="a_22", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
                    "a_23": Task(name="a_23", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),
                    "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND", next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),
                    "a23_notify": Task(name="a23_notify", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_audit": Task(name="a23_audit", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_escalate": Task(name="a23_escalate", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND", merge_from=["a23_notify", "a23_audit", "a23_escalate"], next_tasks=["p1_END"]),

                    "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
                                    next_tasks=["a_8", "a_14"],
                                    conditions=["loop4"]),
                    "p1_END": Task(name="p1_END", next_tasks=[])
                }
            )]
            
        return processes

    if scenario=='scenario_1_B_40_unique': 
        low=0.5
        high=2.0
        # processes = [
        #     ProcessStructure(
        #         name="p_1",
        #         arrival_distribution=l,
        #         data_options={
        #             "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17":0.1},
        #             "priority2": {"p1_END": 0.3, "a_18": 0.7},
        #             "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2":0.1, "a_1":0.2},
        #             "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21":0.1, "a_22":0.2}, 
        #             "request_type": {"a_12": 0.4, "a_13": 0.6},
        #             "loop1": {"a_3": 0.05, "p1_END": 0.95},
        #             "loop2": {"a_16": 0.05, "a_15": 0.95},
        #             "loop3": {"a_10": 0.05, "a_12": 0.95},
        #             "loop4": {"a_8": 0.05, "a_14": 0.95},
        #             "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
        #             "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
        #             "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
        #             "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
        #             "quality_flag": {"q_1": 0.7, "q_2": 0.3},
        #             "report_needed": {"r_1": 0.4, "r_skip": 0.6}
        #         },
        #         tasks={
        #             "START": Task(name="a_start", next_tasks=["a_1"]),
        #             "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),
        #             "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
        #                             next_tasks=["a_2", "a_3", "a_4", "a_17"],
        #                             conditions=["priority"]),
        #             "a_2": Task(name="a_2", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
        #             "a_3": Task(name="a_3", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
        #             "a_3_1": Task(name="a_3_1", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),
        #             "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
        #                             next_tasks=["a_12", "a_6"],
        #                             conditions=["or_sub_1"]),
        #             "a_4": Task(name="a_4", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

        #             "a_5": Task(name="a_5", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
        #             "a_6": Task(name="a_6", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

        #             "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
        #             "a6_diag": Task(name="a6_diag", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
        #             "a6_log": Task(name="a6_log", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
        #             "a6_notify": Task(name="a6_notify", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
        #             "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),


        #             "a_7": Task(name="a_7", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

        #             "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
        #                             next_tasks=["a_8", "a_9", "a_10", "r_2","a_1"],
        #                             conditions=["department"]),

        #             "a_8": Task(name="a_8", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
        #             "a_9": Task(name="a_9", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
        #             "a_10": Task(name="a_10", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),
                    
        #             "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
        #             "a10_qc": Task(name="a10_qc", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
        #             "a10_trace": Task(name="a10_trace", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
        #             "a10_email": Task(name="a10_email", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
        #             "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),





        #             "a_11": Task(name="a_11", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

        #             "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
        #                             next_tasks=["a_12", "a_13"],
        #                             conditions=["request_type"]),
        #             "a_12": Task(name="a_12", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
        #             "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
        #                             next_tasks=["a_12_1", "a_12_2"],
        #                             conditions=["or_sub_3"]),

        #             "a_12_1": Task(name="a_12_1", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
        #             "a_12_2": Task(name="a_12_2", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),




        #             "a_13": Task(name="a_13", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
        #             "a_131": Task(name="a_131", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
        #             "a_132": Task(name="a_132", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
        #             "a_14": Task(name="a_14", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

        #             # New Gateway: Quality Check Branch
        #             "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
        #                 next_tasks=["q_1", "q_2"],
        #                 conditions=["quality_flag"]),

        #             # New Gateway: Escalation Decision
        #             "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
        #                 next_tasks=["e_1", "e_2", "e_3"],
        #                 conditions=["escalation_level"]),

        #             # New Gateway: Reporting Needed
        #             "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
        #                 next_tasks=["r_1", "r_skip"],
        #                 conditions=["report_needed"]),

        #             # New Gateway: Final Loop Check (re-entry)
        #             "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
        #                 next_tasks=["a_3", "p1_END"],
        #                 conditions=["loop1"]),

        #             # New Tasks
        #             "q_1": Task(name="q_1", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
        #             "q_2": Task(name="q_2", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

        #             "e_1": Task(name="e_1", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
        #             "e_2": Task(name="e_2", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
        #             "e_3": Task(name="e_3", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

        #             "r_1": Task(name="r_1", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
        #             "r_skip": Task(name="r_skip", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

        #             "r_2": Task(name="r_2", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

        #             "a_24": Task(name="a_24", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
        #             "a_25": Task(name="a_25", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
        #             "a_26": Task(name="a_26", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

        #             "a_27": Task(name="a_27", resources=[Resource("int19", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
        #             "a_28": Task(name="a_28", resources=[Resource("int19", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
        #             "a_29": Task(name="a_29", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),
                    

        #             "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
        #                             next_tasks=["a_16", "a_15"],
        #                             conditions=["loop2"]),
        #             "a_15": Task(name="a_15", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
        #             "a_16": Task(name="a_16", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),  

        #             "a_17": Task(name="a_17", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),
        #             "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
        #                             next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
        #                             conditions=["or_sub_2"]),
        #             "a_18": Task(name="a_18", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

        #             "a_19": Task(name="a_19", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
        #             "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
        #                                 next_tasks=["a_19_1", "a_19_2", "p1_END"],
        #                                 conditions=["post_processing"]),

        #             "a_19_1": Task(name="a_19_1", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
        #             "a_19_2": Task(name="a_19_2", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),
        #             "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
        #                                 next_tasks=["a_10", "a_12"],
        #                                 conditions=["loop3"]),

        #             "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
        #                                 next_tasks=["p1_END", "a_18"],
        #                                 conditions=["priority2"]),

        #             "a_21": Task(name="a_21", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
        #             "a_22": Task(name="a_22", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
        #             "a_23": Task(name="a_23", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),
        #             "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND", next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),
        #             "a23_notify": Task(name="a23_notify", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
        #             "a23_audit": Task(name="a23_audit", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
        #             "a23_escalate": Task(name="a23_escalate", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
        #             "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND", merge_from=["a23_notify", "a23_audit", "a23_escalate"], next_tasks=["p1_END"]),

        #             "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
        #                             next_tasks=["a_8", "a_14"],
        #                             conditions=["loop4"]),
        #             "p1_END": Task(name="p1_END", next_tasks=[])
        #         }
        #     )]

        processes = [
            ProcessStructure(
                name="p_1",
                arrival_distribution=l,
                data_options={
                    "priority": {"a_2": 0.3, "a_3": 0.3, "a_4": 0.3, "a_17": 0.1},
                    "priority2": {"p1_END": 0.3, "a_18": 0.7},
                    "department": {"a_8": 0.3, "a_9": 0.2, "a_10": 0.2, "r_2": 0.1, "a_1": 0.2},
                    "or_sub_2": {"a_18": 0.3, "a_19": 0.2, "a_5": 0.2, "a_21": 0.1, "a_22": 0.2},
                    "request_type": {"a_12": 0.4, "a_13": 0.6},
                    "loop1": {"a_3": 0.05, "p1_END": 0.95},
                    "loop2": {"a_16": 0.05, "a_15": 0.95},
                    "loop3": {"a_10": 0.05, "a_12": 0.95},
                    "loop4": {"a_8": 0.05, "a_14": 0.95},
                    "or_sub_1": {"a_12": 0.4, "a_6": 0.6},
                    "or_sub_3": {"a_12_1": 0.4, "a_12_2": 0.6},
                    "post_processing": {"a_19_1": 0.5, "a_19_2": 0.3, "p1_END": 0.2},
                    "escalation_level": {"e_1": 0.4, "e_2": 0.3, "e_3": 0.3},
                    "quality_flag": {"q_1": 0.7, "q_2": 0.3},
                    "report_needed": {"r_1": 0.4, "r_skip": 0.6}
                },
                tasks={
                    "START": Task(name="a_start", next_tasks=["a_1"]),

                    # Task order begins
                    "a_1": Task(name="a_1", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_1"]),

                    "XOR_1": Gateway(name="XOR_1", gateway_type="XOR",
                                    next_tasks=["a_2", "a_3", "a_4", "a_17"],
                                    conditions=["priority"]),

                    "a_2": Task(name="a_2", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_5"]),
                    "a_3": Task(name="a_3", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_3_1"]),
                    "a_3_1": Task(name="a_3_1", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_3_1"]),

                    "XOR_3_1": Gateway(name="XOR_3_1", gateway_type="XOR",
                                    next_tasks=["a_12", "a_6"],
                                    conditions=["or_sub_1"]),

                    "a_4": Task(name="a_4", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_7"]),

                    "a_5": Task(name="a_5", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_2"]),
                    "a_6": Task(name="a_6", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["AND_block_1"]),

                    "AND_block_1": Gateway(name="AND_block_1", gateway_type="AND", next_tasks=["a6_diag", "a6_log", "a6_notify"]),
                    "a6_diag": Task(name="a6_diag", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_log": Task(name="a6_log", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "a6_notify": Task(name="a6_notify", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_1"]),
                    "JOIN_block_1": Gateway(name="JOIN_block_1", gateway_type="AND", merge_from=["a6_diag", "a6_log", "a6_notify"], next_tasks=["XOR_2"]),

                    "a_7": Task(name="a_7", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_10"]),

                    "XOR_2": Gateway(name="XOR_2", gateway_type="XOR",
                                    next_tasks=["a_8", "a_9", "a_10", "r_2", "a_1"],
                                    conditions=["department"]),

                    "a_8": Task(name="a_8", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_11"]),
                    "a_9": Task(name="a_9", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_12"]),
                    "a_10": Task(name="a_10", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["AND_block_2"]),

                    "AND_block_2": Gateway(name="AND_block_2", gateway_type="AND", next_tasks=["a10_qc", "a10_trace", "a10_email"]),
                    "a10_qc": Task(name="a10_qc", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "a10_trace": Task(name="a10_trace", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["JOIN_block_2"]),
                    "a10_email": Task(name="a10_email", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_2"]),
                    "JOIN_block_2": Gateway(name="JOIN_block_2", gateway_type="AND", merge_from=["a10_qc", "a10_trace", "a10_email"], next_tasks=["a_11"]),

                    "a_11": Task(name="a_11", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.3))], next_tasks=["XOR_3"]),

                    "XOR_3": Gateway(name="XOR_3", gateway_type="XOR",
                                    next_tasks=["a_12", "a_13"],
                                    conditions=["request_type"]),

                    "a_12": Task(name="a_12", resources=[Resource("int19", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_6"]),
                    "XOR_6": Gateway(name="XOR_6", gateway_type="XOR",
                                    next_tasks=["a_12_1", "a_12_2"],
                                    conditions=["or_sub_3"]),

                    "a_12_1": Task(name="a_12_1", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_14"]),
                    "a_12_2": Task(name="a_12_2", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),

                    # Restart numbering from int1 again
                    "a_13": Task(name="a_13", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_131"]),
                    "a_131": Task(name="a_131", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_132"]),
                    "a_132": Task(name="a_132", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_14"]),
                    "a_14": Task(name="a_14", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_quality"]),

                    "XOR_quality": Gateway(name="XOR_quality", gateway_type="XOR",
                                        next_tasks=["q_1", "q_2"],
                                        conditions=["quality_flag"]),

                    "XOR_escalation": Gateway(name="XOR_escalation", gateway_type="XOR",
                                            next_tasks=["e_1", "e_2", "e_3"],
                                            conditions=["escalation_level"]),

                    "XOR_report": Gateway(name="XOR_report", gateway_type="XOR",
                                        next_tasks=["r_1", "r_skip"],
                                        conditions=["report_needed"]),

                    "XOR_final_loop": Gateway(name="XOR_final_loop", gateway_type="XOR",
                                            next_tasks=["a_3", "p1_END"],
                                            conditions=["loop1"]),

                    "q_1": Task(name="q_1", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_escalation"]),
                    "q_2": Task(name="q_2", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["e_3"]),

                    "e_1": Task(name="e_1", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_report"]),
                    "e_2": Task(name="e_2", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_1"]),
                    "e_3": Task(name="e_3", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["r_2"]),

                    "r_1": Task(name="r_1", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["r_2"]),
                    "r_skip": Task(name="r_skip", resources=[Resource("int11", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_24"]),

                    "r_2": Task(name="r_2", resources=[Resource("int12", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_final_loop"]),

                    "a_24": Task(name="a_24", resources=[Resource("int13", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_25"]),
                    "a_25": Task(name="a_25", resources=[Resource("int14", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_26"]),
                    "a_26": Task(name="a_26", resources=[Resource("int15", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["a_27"]),

                    "a_27": Task(name="a_27", resources=[Resource("int16", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["a_28"]),
                    "a_28": Task(name="a_28", resources=[Resource("int17", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_29"]),
                    "a_29": Task(name="a_29", resources=[Resource("int18", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_4"]),

                    "XOR_4": Gateway(name="XOR_4", gateway_type="XOR",
                                    next_tasks=["a_16", "a_15"],
                                    conditions=["loop2"]),

                    "a_15": Task(name="a_15", resources=[Resource("int19", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_17"]),
                    "a_16": Task(name="a_16", resources=[Resource("int20", (round(random.uniform(low, high), 2), 0.25))], next_tasks=["a_3"]),

                    "a_17": Task(name="a_17", resources=[Resource("int21", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_5"]),

                    "XOR_5": Gateway(name="XOR_5", gateway_type="XOR",
                                    next_tasks=["a_18", "a_19", "a_5", "a_21", "a_22"],
                                    conditions=["or_sub_2"]),

                    "a_18": Task(name="a_18", resources=[Resource("int1", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["p1_END"]),

                    "a_19": Task(name="a_19", resources=[Resource("int2", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["XOR_post_1"]),
                    "XOR_post_1": Gateway(name="XOR_post_1", gateway_type="XOR",
                                        next_tasks=["a_19_1", "a_19_2", "p1_END"],
                                        conditions=["post_processing"]),

                    "a_19_1": Task(name="a_19_1", resources=[Resource("int3", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_post_2"]),
                    "a_19_2": Task(name="a_19_2", resources=[Resource("int4", (round(random.uniform(low, high), 2), 0.15))], next_tasks=["XOR_delete_loop"]),

                    "XOR_delete_loop": Gateway(name="XOR_delete_loop", gateway_type="XOR",
                                            next_tasks=["a_10", "a_12"],
                                            conditions=["loop3"]),

                    "XOR_post_2": Gateway(name="XOR_post_2", gateway_type="XOR",
                                        next_tasks=["p1_END", "a_18"],
                                        conditions=["priority2"]),

                    "a_21": Task(name="a_21", resources=[Resource("int5", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["XOR_a_21"]),
                    "a_22": Task(name="a_22", resources=[Resource("int6", (round(random.uniform(low, high), 2), 0.2))], next_tasks=["a_23"]),
                    "a_23": Task(name="a_23", resources=[Resource("int7", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["AND_block_3"]),

                    "AND_block_3": Gateway(name="AND_block_3", gateway_type="AND",
                                        next_tasks=["a23_notify", "a23_audit", "a23_escalate"]),

                    "a23_notify": Task(name="a23_notify", resources=[Resource("int8", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_audit": Task(name="a23_audit", resources=[Resource("int9", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "a23_escalate": Task(name="a23_escalate", resources=[Resource("int10", (round(random.uniform(low, high), 2), 0.1))], next_tasks=["JOIN_block_3"]),
                    "JOIN_block_3": Gateway(name="JOIN_block_3", gateway_type="AND",
                                            merge_from=["a23_notify", "a23_audit", "a23_escalate"],
                                            next_tasks=["p1_END"]),

                    "XOR_a_21": Gateway(name="XOR_a_21", gateway_type="XOR",
                                        next_tasks=["a_8", "a_14"],
                                        conditions=["loop4"]),

                    "p1_END": Task(name="p1_END", next_tasks=[])
                }
            )
        ]

            
        return processes
    
    if scenario == 'bpi2020_2processes_massive_share':
        processes = [
            ProcessStructure(
                name="1_payment_request",
                arrival_distribution=l,
                data_options={
                    "1_approval_flow": {"1_submit_request": 0.15, "1_check_budget_approval": 0.85},
                    "1_budget_check": {"1_budget_approve": 0.35, "1_supervisor_approve": 0.65},
                    "1_final_payment_check": {"1_handle_payment": 0.01, "1_END": 0.99}
                },
                tasks={
                    "START": Task(name="1_start", next_tasks=["1_submit_request"]),

                    # ---- Process 1 (FASTER process) ----
                    # Early tasks: dedicated ≈ 0.95–1.10, X_SHARED = 0.75  (very attractive)
                    "1_submit_request": Task(name="1_submit_request", resources=[
                        Resource("1employee", (0.95, 0.1)),
                        Resource("1employee1", (0.75, 0.1)),
                    ], next_tasks=["1_check_request"]),

                    "1_check_request": Task(name="1_check_request", resources=[
                        Resource("112admin0", (1.10, 0.1)),
                        Resource("12admin1", (1.20, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_XOR_admin_decision"]),

                    "1_XOR_admin_decision": Gateway(name="1_XOR_admin_decision", gateway_type="XOR",
                                                    next_tasks=["1_submit_request", "1_check_budget_approval"],
                                                    conditions=["1_approval_flow"]),

                    "1_check_budget_approval": Task(name="1_check_budget_approval", resources=[
                        Resource("12admin0", (1, 0.1)),
                        Resource("112admin1", (1.10, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_XOR_budget_check"]),

                    "1_XOR_budget_check": Gateway(name="1_XOR_budget_check", gateway_type="XOR",
                                                next_tasks=["1_budget_approve", "1_supervisor_approve"],
                                                conditions=["1_budget_check"]),

                    "1_budget_approve": Task(name="1_budget_approve", resources=[
                        Resource("1budget_owner1", (1.05, 0.1)),
                        Resource("1budget_owner2", (1.2, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_supervisor_approve"]),

                    "1_supervisor_approve": Task(name="1_supervisor_approve", resources=[
                        Resource("1supervisor1", (1.1, 0.1)),
                        Resource("1supervisor2", (1.20, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_handle_payment"]),

                    # Final task: dedicated VERY SLOW (3.2/3.8), X_SHARED = 1.25 (best but not the fastest overall)
                    "1_handle_payment": Task(name="1_handle_payment", resources=[
                        Resource("12accounting0", (3.20, 0.1)),
                        Resource("12accounting1", (3.80, 0.1)),
                        Resource("X_SHARED", (1., 0.1)),
                    ], next_tasks=["1_XOR_payment_check"]),

                    "1_XOR_payment_check": Gateway(name="1_XOR_payment_check", gateway_type="XOR",
                                                next_tasks=["1_handle_payment", "1_END"],
                                                conditions=["1_final_payment_check"]),

                    "1_END": Task(name="1_END", next_tasks=[])
                }
            ),

            ProcessStructure(
                name="2_payment_request",
                arrival_distribution=l,
                data_options={
                    "2_approval_flow": {"2_submit_request": 0.15, "2_check_budget_approval": 0.85},
                    "2_budget_check": {"2_budget_approve": 0.35, "2_supervisor_approve": 0.65},
                    "3_final_payment_check": {"2_handle_payment": 0.01, "2_END": 0.99}
                },
                tasks={
                    "START": Task(name="2_start", next_tasks=["2_submit_request"]),

                    # ---- Process 2 (SLOWER process) ----
                    # Early tasks: dedicated ≈ 1.30–1.50, X_SHARED = 0.95 (still attractive)
                    "2_submit_request": Task(name="2_submit_request", resources=[
                        Resource("1employee", (1.30, 0.1)),
                        Resource("1employee1", (0.95, 0.1)),
                    ], next_tasks=["2_check_request"]),

                    "2_check_request": Task(name="2_check_request", resources=[
                        Resource("112admin0", (1.30, 0.1)),
                        Resource("12admin1", (1.60, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_XOR_admin_decision"]),

                    "2_XOR_admin_decision": Gateway(name="2_XOR_admin_decision", gateway_type="XOR",
                                                    next_tasks=["2_submit_request", "2_check_budget_approval"],
                                                    conditions=["2_approval_flow"]),

                    "2_check_budget_approval": Task(name="2_check_budget_approval", resources=[
                        Resource("12admin0", (1.30, 0.1)),
                        Resource("112admin1", (1.40, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_XOR_budget_check"]),

                    "2_XOR_budget_check": Gateway(name="2_XOR_budget_check", gateway_type="XOR",
                                                next_tasks=["2_budget_approve", "2_supervisor_approve"],
                                                conditions=["2_budget_check"]),

                    "2_budget_approve": Task(name="2_budget_approve", resources=[
                        Resource("1budget_owner1", (1.30, 0.1)),
                        Resource("1budget_owner2", (1.5, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_supervisor_approve"]),

                    "2_supervisor_approve": Task(name="2_supervisor_approve", resources=[
                        Resource("1supervisor1", (1.30, 0.1)),
                        Resource("1supervisor2", (1.5, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_handle_payment"]),

                    # Final task: dedicated EVEN SLOWER (4.4/5.0), X_SHARED = 1.25 (best)
                    "2_handle_payment": Task(name="2_handle_payment", resources=[
                        Resource("12accounting0", (4.40, 0.1)),
                        Resource("12accounting1", (3.00, 0.1)),
                        Resource("1X_SHARED", (1, 0.1)),
                    ], next_tasks=["2_XOR_payment_check"]),

                    "2_XOR_payment_check": Gateway(name="2_XOR_payment_check", gateway_type="XOR",
                                                next_tasks=["2_handle_payment", "2_END"],
                                                conditions=["3_final_payment_check"]),

                    "2_END": Task(name="2_END", next_tasks=[])
                }
            ),
        ]
        return processes



       
 


ARRIVAL_RATES= [.5]#not used
SCENARIO_NAMES = ['scenario_1_B_40_unique']#['scenario_1_A', 'scenario_1_B_75_unique',  'scenario_1_B_40_unique', 'scenario_1_B_20_unique', 'scenario_1_B', 'bpi2020_2processes_massive_share']
SIMULATION_RUN_TIME =  5000#5000 for all scenarios #10000 for bpi2020_2processes_massive_share
SIMULATION_RUNS = 1 #5





























