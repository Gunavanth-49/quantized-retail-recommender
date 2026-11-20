"""Streamlit entry point: streamlit run app/main.py."""

import streamlit as st

from app.core import (
    CATALOG,
    checkpoint_bytes,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)


@st.cache_resource(show_spinner="Training and evaluating basket models…")
def run_experiment():
    splits = milestone_1()
    source, fp32 = milestone_2(splits)
    baseline = milestone_4(fp32, splits["test"])
    compressed = milestone_5(fp32, splits["calibration"])
    comparison = milestone_6(baseline, compressed, splits["test"])
    return (
        splits, {"FP32": fp32, **compressed}, comparison,
        milestone_4(source, splits["test"]),
    )


def main():
    st.set_page_config(page_title="Quantized Retail Recommender", layout="wide")
    st.title("Quantized Retail Recommender")
    splits, models, comparison, source_metrics = run_experiment()
    recommend, benchmarks = st.tabs(["Basket recommendations", "Model comparison"])

    with recommend:
        basket = st.multiselect("Items in your basket", CATALOG, default=["Bread", "Butter"])
        variant = st.selectbox("Model", models.keys(), index=1)
        if basket:
            result = milestone_3(models[variant], basket, splits["train"])
            st.metric("Training basket support", f'{result["support"]:.1%}')
            st.caption(f'{result["matches"]:,} of {len(splits["train"]):,} training baskets contain all selected items.')
            if result["recommendations"]:
                st.table([
                    {
                        "Item": row["item"],
                        "Model probability": f'{row["probability"]:.1%}',
                        "Co-purchase confidence": (
                            f'{row["confidence"]:.1%}' if row["confidence"] is not None else "—"
                        ),
                    }
                    for row in result["recommendations"]
                ])
            else:
                st.info("All catalog items are already in the basket.")
        else:
            st.info("Select at least one item to see recommendations.")

    with benchmarks:
        st.caption(f'Each model was evaluated on the same {comparison["FP32"]["examples"]:,} held-out next-item examples.')
        st.table([
            {
                "Model": name, "Checkpoint bytes": row["bytes"],
                "Size reduction": f'{row["reduction_pct"]:.1f}%',
                "Top-1": f'{row["top_1"]:.1%}', "Top-5": f'{row["top_5"]:.1%}',
                "Top-1 change (pp)": f'{row["top_1_delta_pp"]:+.2f}',
                "Top-5 change (pp)": f'{row["top_5_delta_pp"]:+.2f}',
            }
            for name, row in comparison.items()
        ])
        st.caption(f'Source checkpoint before continued pretraining: {source_metrics["top_1"]:.1%} top-1 on the same holdout.')
        download_variant = st.selectbox("Checkpoint to download", models.keys())
        st.download_button(
            "Download NPZ checkpoint",
            checkpoint_bytes(models[download_variant]),
            file_name=download_variant.lower().replace(" ", "-") + ".npz",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    main()
