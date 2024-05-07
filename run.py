from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "generative-ai-authorship-verification-panclef-2024",
        "pan24-generative-authorship-tiny-smoke-20240417-training",
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")

    pred1 = model.predict_proba(df["text1"])
    pred2 = model.predict_proba(df["text2"])
    p_text_1_llm = pred1[:, 1]
    p_text_2_human = pred2[:, 0]

    df["is_human"] = p_text_1_llm + p_text_2_human / 2
    df = df[["id", "is_human"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "naive.jsonl", orient="records", lines=True)
