import numpy as np
from collections import defaultdict
from healthcare_sim.config import NUM_STEPS
import copy

def run_simulation(Patient, patients, pathways, actions, OUTPUT_ACTIONS, INPUT_ACTIONS,
        NUM_PATHWAYS, NUM_STEPS, IDEAL_CLINICAL_VALUES):
    from healthcare_sim.action import Action
    import time

    print("Running simulation...")
    start_time = time.time()
    system_cost = {}
    sum_cost = 0
    activity_log = []

    for step in range(NUM_STEPS):
        step_cost = 0
        rewards = []

        system_state = 0  # calculste the total queue

        for p in patients:
            Patient.clinical_decay(p, IDEAL_CLINICAL_VALUES) 
            queue_penalty = p.queue_time ** 2  # Quadratic penalty
            clinical_penalty = np.exp(p.outcomes['clinical_penalty'] / 50) # Exponential penalty
            action_cost = actions[next_a].cost if next_a in actions else 0
            reward = - 0.25 * action_cost - 0.5 * clinical_penalty - 0.0001 * queue_penalty - 0.5 * system_state 
            rewards.append(reward)

            #create the qubo

            #call the simulator

            

            if next_a == OUTPUT_ACTIONS:
                        if pw.name in p.diseases:
                            p.diseases[pw.name] = False 
            
            for act in actions.values():
                in_progress, cost = act.execute(IDEAL_CLINICAL_VALUES,)
                step_cost += cost
            sum_cost += step_cost
            system_cost[step] = sum_cost

        end_time = time.time()
        print(f"Run completed in {end_time - start_time:.2f} seconds")   
    return system_cost, activity_log