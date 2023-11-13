import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

hue_order = None
for track in ["sub"]:

    if track == "sub":
        dataframe = f"/home/samco/dgx/CHiME6/test_chime7/espnet/egs2/chime7_task1/asr1/scoring_participants_2/{track}_macro_avg.csv"
    else:
        dataframe = f"/home/samco/dgx/CHiME6/test_chime7/espnet/egs2/chime7_task1/asr1/scoring_participants/{track}_scn.csv"
    dataframe = pd.read_csv(dataframe)

    # select
    #c_partition_df = dataframe[dataframe["partition"] == partition]

    for partition in ["dev", "eval"]:
        c_partition_df = dataframe[dataframe["partition"] == partition]
        # c_partition_df = c_partition_df[c_partition_df["team"] != "Padeborn"]
        c_partition_df = c_partition_df.sort_values('wer',
                                                    ascending=True).groupby(
            'team').head(1)
        if hue_order is None:
            hue_order = c_partition_df.sort_values("team")["team"].tolist()
            if "Padeborn" in hue_order:
                hue_order.remove("Padeborn")
                hue_order.append("Padeborn")
        if track == "sub" and partition == "dev":
            hue_order.remove("Padeborn")
            hue_order.append("BUT")

        if track == "a":
            fig, ax = plt.subplots()
            ax.scatter(c_partition_df["diarization error rate"],
                       c_partition_df["wer"])

            for i, txt in enumerate(c_partition_df["team"]):
                ax.annotate(txt,
                            (c_partition_df["diarization error rate"].iloc[i],
                             c_partition_df["wer"].iloc[i]))

            plt.xlabel("DER")
            plt.ylabel("DA-WER")
            plt.tight_layout()
            plt.savefig(f"/tmp/{track}_{partition}.png")
        else:

            plt.figure()
            sns.barplot(x="team",
                        y="wer",
                        data=c_partition_df, hue="team", hue_order=hue_order,
                        dodge=False)

            plt.legend([], [], frameon=False)
            plt.xticks(rotation=70)

            plt.title(f"{track} track, {partition} partition")
            plt.xlabel("Team Name")
            plt.ylabel("DA-WER")
            plt.tight_layout()
            plt.savefig(f"/tmp/{track}_{partition}.png")







