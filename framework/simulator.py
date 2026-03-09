from enum import Enum, auto
from typing import Optional, Dict
import random
import pandas as pd
import heapq
from framework.processes import ProcessElement, Resource, Process

class SimulationItemType(Enum):
    CASE_ARRIVAL = auto()
    ACTIVATE_TASK = auto()
    COMPLETE_TASK = auto()
    COMPLETE_EVENT = auto()
    COMPLETE_CASE = auto()
    ASSIGN_RESOURCES = auto()
    START_TASK = auto()

class SimulationItem:
    def __init__(self, simulation_item_type: SimulationItemType, moment: float, process_element: Optional[ProcessElement], resource: Optional[Resource] = None):
        self.simulation_item_type = simulation_item_type
        self.moment = moment
        self.process_element = process_element
        self.resource = resource

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return (
            f"{self.simulation_item_type} | time:{round(self.moment, 2)} | "
            f"{self.process_element} | {self.resource}"
        )

class EventLog:
    def __init__(self):
        self.log = []

    def log_event(self, **kwargs):
        self.log.append(kwargs)

    def get_dataframe(self):
        
        return pd.DataFrame(self.log)
class Simulator:
    def __init__(self, simulation_run: int, process: Process):
        self.simulation_run = simulation_run
        self.process = process
        self.now = 0
        self.events = []
        self.unassigned_tasks = {}
        self.busy_resources = {}
        self.available_resources = set()
        self.case_start_times = {}
        self.task_start_times = {}
        self.busy_cases = {}
        self.finalized_cases = 0
        self.total_cycle_time = 0
        self.event_log = EventLog()
        print("[Simulator Initialized]")
        process.set_simulator(self)
        
        self.initialize_simulation()

    def initialize_simulation(self):
        #print("[Simulation Initialization]")
        self.available_resources = set(self.process.resources)
        #print(f"[Available Resources Initialized] {self.available_resources}")
        self.process.reset_all_process_parameter()
        initial_time, initial_event = self.process.next_case()
        #print(f"[Initial Event Scheduled] Time: {initial_time}, Event: {initial_event}")
        self.schedule_event(initial_time, SimulationItem(simulation_item_type=SimulationItemType.CASE_ARRIVAL, moment=initial_time, process_element=initial_event))
        #print("[Simulation Initialization End]")


    def schedule_event(self, moment: float, simulation_item: SimulationItem):
        #print(f"[Event Scheduled] Time: {moment}, Event: {simulation_item} ---------------------")
        #self.events.append((moment, simulation_item))
        heapq.heappush(self.events, (moment, simulation_item))
        #self.events.sort()

    def run(self, running_time: float):
        #print(f"[Simulation Start] Running Time: {running_time}")
        while self.now <= running_time:# and self.events:
            #self.now, current_event = self.events.pop(0)
            self.now, current_event = heapq.heappop(self.events)
            #print(f"[Processing Event] Time: {self.now}, Event: {current_event}")
            if current_event.process_element:
                print(round(self.now,2)," | ",current_event.process_element.case_type," | ", current_event.process_element.case_id," | ",current_event.process_element.id," | ", current_event.process_element.label," | ", current_event.simulation_item_type," | ", current_event.resource, " | ", self.finalized_cases)
            else:
                print(round(self.now,2)," | ", "---"," | ", "---"," | ", current_event.simulation_item_type," | ", current_event.resource)
            self.handle_event(current_event)
            #self.sort_events()
        #print(f"[Simulation End] Finalized Cases: {self.finalized_cases}, Total Cycle Time: {self.total_cycle_time}")


    def handle_event(self, simulation_item: SimulationItem):
        #print(f"[Handle Event] Type: {simulation_item.simulation_item_type}, Element: {simulation_item.process_element}")
        process_element = simulation_item.process_element

        if simulation_item.simulation_item_type == SimulationItemType.CASE_ARRIVAL:
            self.handle_case_arrival(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_EVENT:
            self.handle_complete_event(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_TASK:
            self.handle_complete_task(simulation_item)

        elif simulation_item.simulation_item_type == SimulationItemType.START_TASK:
            self.handle_start_task(simulation_item)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_CASE:
            self.handle_complete_case(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.ASSIGN_RESOURCES:
            self.handle_assign_resources()

    def handle_case_arrival(self, process_element: ProcessElement):
        #print(f"[Case Arrival] Case ID: {process_element.case_id}, Time: {self.now}")
        case_id = process_element.case_id
        self.case_start_times[case_id] = self.now
        self.busy_cases[case_id] = []
        self.activate_element(process_element)
        next_time, next_event = self.process.next_case()
        #print(f"[Next Case Scheduled] Time: {next_time}, Event: {next_event}")
        self.schedule_event(next_time, SimulationItem(SimulationItemType.CASE_ARRIVAL, next_time, next_event))
        self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "START", 
                                 case_id=case_id,
                                 activity=process_element.label,
                                 data=self.process.case_data[process_element.case_id])
        
    def handle_complete_event(self, process_element: ProcessElement):
        #print(f"[Complete Event] Element: {process_element}")
        #self.busy_cases[process_element.case_id].remove(process_element)
        next_elements = self.process.complete_element(process_element)
        if not next_elements:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.COMPLETE_CASE, self.now, process_element))
        elif process_element.is_gateway:
            self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "gateway", 
                                 case_id=process_element.case_id,
                                 activity=process_element.label,
                                 data=self.process.case_data[process_element.case_id])
        for next_element in next_elements:
            self.activate_element(next_element)
        
        

    def handle_complete_task(self, simulation_item: SimulationItem):
        #print(f"[Complete Task] Resource: {simulation_item.resource}, Element: {simulation_item.process_element}")
        resource = simulation_item.resource
        process_element = simulation_item.process_element

        next_elements = self.process.complete_element(process_element)

        self.busy_resources.pop(resource, None)
        self.available_resources.add(resource)
        #self.busy_cases[process_element.case_id].remove(process_element)

        
        
        if not next_elements:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.COMPLETE_CASE, self.now, process_element))
        
        for next_element in next_elements:
            self.activate_element(next_element)

        self.schedule_event(self.now, SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None))


    def handle_start_task(self, simulation_item: SimulationItem):
        #print(f"[Start Task] Resource: {simulation_item.resource}, Element: {simulation_item.process_element}")
        resource = simulation_item.resource
        process_element = simulation_item.process_element
        case_id = process_element.case_id
        self.busy_resources[resource] = process_element
        processing_time = self.process.processing_time_sample(resource, process_element, self.now)
        self.task_start_times[process_element.id] = self.now
        #print(f"[Task Processing Time] Resource: {resource}, Time: {processing_time}")
        self.schedule_event(self.now + processing_time, SimulationItem(SimulationItemType.COMPLETE_TASK, self.now + processing_time, process_element, resource))
        self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "running", 
                                 case_id=case_id,
                                 activity=process_element.label,
                                 resource=resource.name,
                                 end_time=self.now + processing_time)

    def handle_complete_case(self, process_element: ProcessElement):
        #print(f"[Complete Case] Case ID: {process_element.case_id}")
        case_id = process_element.case_id
        is_gateway = process_element.is_gateway
        self.finalized_cases += 1
        cycle_time = self.now - self.case_start_times[case_id]
        self.total_cycle_time += cycle_time
        if is_gateway:
            status='gateway'
        else:
            status='COMPLETE'
        self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = status, 
                                 case_id=case_id, 
                                 activity=process_element.label,
                                 cycle_time=cycle_time)

    def handle_assign_resources(self):
        #print("[Assign Resources]")
        assignments = self.process.assign_resources(self.unassigned_tasks, self.available_resources)
        for task, resource in assignments:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.START_TASK, self.now, task, resource))
            self.unassigned_tasks.pop(task.id, None)
            self.available_resources.remove(resource)

    def activate_element(self, process_element: ProcessElement):
        #print(f"[Activate Element] Element: {process_element}")
        self.busy_cases[process_element.case_id].append(process_element)
        if process_element.is_event():
            self.schedule_event(process_element.occurrence_time, SimulationItem(SimulationItemType.COMPLETE_EVENT, process_element.occurrence_time, process_element))
        elif process_element.is_task():
            self.unassigned_tasks[process_element.id] = process_element
            self.schedule_event(self.now, SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None))

    # def sort_events(self) -> None:
    #     """First start tasks (i.e. use resources) before another COMPLETE_EVENT comes into action"""
    #     self.events.sort(key = lambda k : (k[0], # time
	# 								 1 if k[1].simulation_item_type == SimulationItemType.COMPLETE_EVENT else
	# 								 0)
	# 	)
