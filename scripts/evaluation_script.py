import run_profiler_buysell
import run_baseline_buysell

if __name__ == "__main__":
    
    experiment_type = int(input("Which experiment would you like to run? (1) Baseline, (2) Profiler: "))

    if experiment_type == 1:
        log_iteration = int(input("What iteration of this run are you on? (This number should be a multiple of 30) : "))
        run_baseline_buysell.run_buysell_experiment(
            log_iteration=log_iteration,
            self_is_buyer = False,
            opponent_model="api-gpt-oss-120b",
            self_model="meta-llama/Meta-Llama-3-8B-Instruct",
            seller_cost=40,
            buyer_wtp=60,
            iterations=10,
            max_retries=3,
            temperature=0.3,
            max_tokens=800,
        
        )

    else:
        log_iteration = int(input("What iteration of this run are you on? (This number should be a multiple of 30) : "))
        print(log_iteration)
        run_profiler_buysell.run_profiler_experiment(
            
            log_iteration = log_iteration,
            opponent_model="api-gpt-oss-120b",
            self_model="meta-llama/Meta-Llama-3-8B-Instruct",
            profiler_model = "api-gpt-oss-120b",
            seller_cost = 40, 
            buyer_wtp = 60, 
            iterations= 10, 
            max_retries=3, 

        )
    


