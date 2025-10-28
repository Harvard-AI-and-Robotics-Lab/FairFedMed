import os
import re
import numpy as np
import pandas as pd


def main(root_folder, epoch=49, num_clients=3):
    current_path = os.getcwd()
    root_folder = os.path.join(current_path, root_folder)

    # Match client section header
    client_header_pattern = re.compile(r"Evaluate on the (client\d+)_test set")
    # Match metrics lines
    metric_line_pattern = re.compile(r"\* ([\w\d_]+): ([\d.]+)%?")

    # Store metrics as: {client_id: {metric_name: [values across logs]}}
    client_metrics = {}

    # Parse metrics for epoch 49
    def extract_epoch49_client_metrics(log_content, epoch):
        lines = log_content.splitlines()
        in_epoch49 = False
        current_client = None
        parsed_data = {}

        for line in lines:
            if "local train finish epoch:" in line and str(epoch) in line:
                in_epoch49 = True
            elif in_epoch49 and "local train finish epoch:" in line:
                break  # next epoch begins, stop parsing
            elif in_epoch49:
                client_match = re.search(client_header_pattern, line)
                if client_match:
                    current_client = client_match.group(1)
                    if current_client not in parsed_data:
                        parsed_data[current_client] = {}
                elif current_client and line.strip().startswith("*"):
                    metric_match = re.match(metric_line_pattern, line.strip())
                    if metric_match:
                        metric_name = metric_match.group(1)
                        value = float(metric_match.group(2))
                        if metric_name not in parsed_data[current_client]:
                            parsed_data[current_client][metric_name] = []
                        parsed_data[current_client][metric_name].append(value)
        return parsed_data

    # Go through each folder and collect metrics
    for i, folder_name in enumerate(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        log_file_path = os.path.join(folder_path, "log.txt")
        if os.path.isdir(folder_path) and os.path.isfile(log_file_path):
            with open(log_file_path, "r") as f:
                content = f.read()
                parsed = extract_epoch49_client_metrics(content, epoch-i)
                for client_id, metrics in parsed.items():
                    if client_id not in client_metrics:
                        client_metrics[client_id] = {}
                    for metric_name, values in metrics.items():
                        if metric_name not in client_metrics[client_id]:
                            client_metrics[client_id][metric_name] = []
                        client_metrics[client_id][metric_name].extend(values)

    # Compute mean and std for each client-metric
    records = []
    for client_id, metrics in client_metrics.items():
        for metric_name, values in metrics.items():
            records.append({
                "client": client_id,
                "metric": metric_name,
                "mean": np.mean(values),
                "std": np.std(values)
            })

    df_result = pd.DataFrame(records)
    # print(df_result)
    df_result.to_csv(f"{root_folder}/epoch49_client_metrics.csv", index=False)

    # Cross-client aggregation
    client_avg_df = df_result.groupby("metric").agg({
        "mean": "mean",
        "std": "mean"  # 这里 std 也取平均值（跨 client 的 std 均值）
    }).reset_index()
    client_avg_df["client"] = "client_avg"

    # 合并并保存
    df_combined = pd.concat([df_result, client_avg_df], ignore_index=True)
    df_combined.to_csv(f"{root_folder}/epoch49_client_metrics_with_avg.csv", index=False)

    # # 打印预览
    # print(df_combined[df_combined["client"] == "client_avg"])

    # Adaptively extract and format auc_{attr}_* metrics based on actual data
    for i in range(1, len(root_folder.split("/")[-2].split("_"))):
        attr = root_folder.split("/")[-2].split("_")[-i]
        if attr in ["ethnicity", "gender", "language", "race", "age"]:
            break

    def filter_and_format_adaptive(df, attr, num_clients):
        result = []
        if num_clients == 3:
            ordered_clients = ['client0', 'client1', 'client2', 'client_avg']
        elif num_clients == 2:
            ordered_clients = ['client0', 'client1', 'client_avg']
        else:
            raise NotImplementedError
        
        # Find available auc_{attr}_* keys
        auc_keys = df[df["metric"].str.startswith(f"auc_{attr}_")]["metric"].unique()
        auc_keys_sorted = sorted(auc_keys, key=lambda x: int(x.split("_")[-1]))  # Sort by index
        
        # Final metric order
        ordered_metrics = ['overall_auc', f'esauc_{attr}'] + auc_keys_sorted + [f'eod_{attr}',  f'dpd_{attr}']
        result.append(" & ".join(ordered_metrics))

        for client in ordered_clients:
            subdf = df[(df["client"] == client) & (df["metric"].isin(ordered_metrics))]
            subdf = subdf.set_index("metric").reindex(ordered_metrics)
            values = [
                f'{m:.1f} ($\pm$ {s:.1f})' for m, s in zip(subdf["mean"], subdf["std"])
            ]
            line = f"{client}: " + " & ".join(values)
            result.append(line)

        return result

    adaptive_formatted_outputs = filter_and_format_adaptive(df_combined, attr, num_clients)
    print()
    print(root_folder)
    print(adaptive_formatted_outputs)

    # Save to file
    output_path = f"{root_folder}/epoch49_{attr}_metrics.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(adaptive_formatted_outputs))  

epoch = 49
num_clients=3

# rn50_oph
# root_folders = [
#     # PromptFL
#     "output/PromptFL_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/PromptFL_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/PromptFL_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/PromptFL_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FedOTP
#     "output/FedOTP_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/FedOTP_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/FedOTP_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/FedOTP_rn50_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FairLoRA
#     "output/FairLoRA_rn50_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_ethnicity_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_rn50_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_rn50_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_language_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_rn50_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_onehotattr0.7/nctx4_cscFalse_ctpend",
# ]   

# # vit_oph
# root_folders = [
#     # PromptFL
#     "output/PromptFL_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FedOTP
#     "output/FedOTP_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph/fairfedmed_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FairLoRA
#     "output/FairLoRA_vit_b16_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_ethnicity_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_language_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_ema/fairfedmed_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
# ]   

epoch = 29
# rn50_oph oct
root_folders = [
    # # PromptFL
    "output/PromptFL_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
    "output/PromptFL_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
    "output/PromptFL_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
    "output/PromptFL_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
    # # FedOTP epoch = 29
    # "output/FedOTP_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
    # "output/FedOTP_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
    # "output/FedOTP_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
    # "output/FedOTP_rn50_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
    # # FairLoRA
    # "output/FairLoRA_rn50_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_itv4_slice8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_ethnicity_onehotattr0.7/nctx4_cscFalse_ctpend",
    # "output/FairLoRA_rn50_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_itv4_slice8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_onehotattr0.7/nctx4_cscFalse_ctpend",
    # "output/FairLoRA_rn50_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_itv4_slice8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_language_onehotattr0.7/nctx4_cscFalse_ctpend",
    # "output/FairLoRA_rn50_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank32_alpha8_itv4_slice8_sinit_half_half_cycle0.1g/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_onehotattr0.7/nctx4_cscFalse_ctpend",
]   

# root_folders = [
#     # PromptFL
#     "output/PromptFL_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/PromptFL_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FedOTP epoch = 29
#     "output/FedOTP_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_ethnicity/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_language/nctx4_cscFalse_ctpend",
#     "output/FedOTP_vit_b16_oph_oct/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_itv4_slice8/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     # FairLoRA
#     "output/FairLoRA_vit_b16_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_itv4_slice8/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_ethnicity_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_itv4_slice8/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_itv4_slice8/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_language_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/FairLoRA_vit_b16_oph_oct_ema/fairfedmed_oct_bscans_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_itv4_slice8/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_onehotattr0.7/nctx4_cscFalse_ctpend",
# ]

# fedchexmimic
# num_clients = 2
# root_folders = [
#     # PromptFL
#     "output/fedchexmimic/PromptFL_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/PromptFL_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/PromptFL_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_age/nctx4_cscFalse_ctpend",
#     # FedOTP
#     "output/fedchexmimic/FedOTP_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FedOTP_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FedOTP_rn50_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_age/nctx4_cscFalse_ctpend",
    # FairLoRA
    # "output/fedchexmimic/FairLoRA_rn50_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
    # "output/fedchexmimic/FairLoRA_rn50_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
    # "output/fedchexmimic/FairLoRA_rn50_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_age_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend"
# ]

# root_folders = [
#     # PromptFL
#     "output/fedchexmimic/PromptFL_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/PromptFL_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/PromptFL_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/PromptFL_GLP_OT_None_0.8_eps0.1_age/nctx4_cscFalse_ctpend",
#     # FedOTP
#     "output/fedchexmimic/FedOTP_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_race/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FedOTP_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_gender/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FedOTP_vit_b16_oph/fedchexmimic_noniid-labeldir100_beta0.3_normalize/FedOTP_GLP_OT_COT_0.8_eps0.1_age/nctx4_cscFalse_ctpend",
#     # FairLoRA
#     "output/fedchexmimic/FairLoRA_vit_b16_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_race_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FairLoRA_vit_b16_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_gender_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend",
#     "output/fedchexmimic/FairLoRA_vit_b16_oph_ema/fedchexmimic_noniid-labeldir100_beta0.3_FairLoRA_rank12_alpha2_sinit_cycle_shift/FedOTPLoRA_GLP_OT_SVLoRA_None_0.8_eps0.1_age_lambda_fairness0.0_onehotattr0.7/nctx4_cscFalse_ctpend"
# ]

for root_folder in root_folders:
    main(root_folder, epoch=epoch, num_clients=num_clients)