<div align="center">

# Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification.

</div>


This repo contains the resources for the paper "Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification."


In this work, we study both sequential and parallel test-time scaling of deep search agents, motivated by the intuition that verification in this setting is often much easier than generation. In experiments, we first show that sequential scaling methods, such as budget forcing, can be effective initially but eventually degrade performance when over-applied in
agentic search. Due to asymmetric verification, however, we are able to achieve substantial improvements by allocating only a modest amount of compute to the verifier. 

<img width="10761" height="6394" alt="Figure1-30_11_00" src="https://github.com/user-attachments/assets/5b668668-1557-43e6-8b49-73780a346667" />


# Quick Start

## Search

```
python3 scripts/deep_search.py \
    --input_path ./data/BrowseCompEN-Sample100/all_data_random100_sample1.json \ # data path
    --output_dir ./outputs/BrowseCompEN-Sample100/qwen3-235b-a22b-2507_main_kimi-k2_aux/max_search_calls_15/parallel_sample \ # ouput path
    --use_aihubmix \
    --aihubmix_api_url "https://openrouter.ai/api/v1/chat/completions" \ # openrouter url or aihubmix url
    --aihubmix_api_keys "" \ # your openrouter keys or aihubmix keys
    --price_config_path "./model_config/openrouter_price.json" \
    --use_google_pro \
    --google_pro_api_key "" \
    --use_custom_api \
    --aux_model_name "moonshotai/kimi-k2" \
    --max_tokens 16384 \
    --model_name "qwen/qwen3-235b-a22b-2507" \
    --top_k 10 \
    --concurrent_limit 100 \
    --max_search_calls 15 \
    --mode "solve" \
```


## Search using Budget Forcing

```
python3 scripts/deep_search.py \
    --input_path ./outputs/policy_results_08-01_02-54-47.json \  # data path, produce by search
    --output_dir ./outputs/solve_budget_forcing_results \ # ouput path
    --use_aihubmix \
    --aihubmix_api_url "https://openrouter.ai/api/v1/chat/completions" \ # openrouter url or aihubmix url
    --aihubmix_api_keys "" \ # your openrouter keys or aihubmix keys
    --price_config_path "./model_config/openrouter_price.json" \
    --use_google_pro \
    --google_pro_api_key  \
    --use_custom_api \
    --aux_model_name "moonshotai/kimi-k2" \
    --max_tokens 16384 \
    --model_name "qwen/qwen3-235b-a22b-2507" \
    --top_k 10 \
    --concurrent_limit 100 \
    --max_search_calls 15 \
    --mode "solve_budget_forcing" \
```


## Verify 

```
python3 scripts/deep_search.py \
    --input_path ./outputs/policy_results_08-01_02-54-47.json \ # data path, produce by search
    --output_dir ./outputs/verify_results \ # ouput path
    --use_aihubmix \
    --aihubmix_api_url "https://openrouter.ai/api/v1/chat/completions" \ # openrouter url or aihubmix url
    --aihubmix_api_keys "" \  # your openrouter keys or aihubmix keys
    --price_config_path "./model_config/openrouter_price.json" \
    --use_google_pro \
    --google_pro_api_key "" \
    --use_custom_api \
    --aux_model_name "moonshotai/kimi-k2" \
    --max_tokens 16384 \
    --model_name "qwen/qwen3-235b-a22b-2507" \
    --top_k 10 \
    --concurrent_limit 100 \
    --max_search_calls 15 \
    --mode "verify" \
```
