import re
import pandas as pd

# Every question in the source dataset starts with one of these instruction
# templates (it was built for instruction-tuning). Left in, they dominate the
# embeddings and make all 18k questions look alike to the retriever.
INSTRUCTION_PREFIXES = [
    "utilize your financial knowledge, give your answer or opinion to the input question or subject . answer format is not limited.",
    "based on your financial expertise, provide your response or viewpoint on the given financial question or topic. the response format is open.",
    "offer your thoughts or opinion on the input financial query or topic using your financial background.",
    "share your insights or perspective on the financial matter presented in the input.",
    "offer your insights or judgment on the input financial query or topic using your financial expertise. reply as normal question answering",
]
_PREFIX_RE = re.compile(r"^(" + "|".join(re.escape(p) for p in INSTRUCTION_PREFIXES) + r")\s*")


def strip_prefix(text: str) -> str:
    return _PREFIX_RE.sub("", text).strip()


def load_corpus(path: str, clean: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns question, answer."""
    df = pd.read_csv(path).dropna()
    df = df.rename(columns={"input": "question", "output": "answer"})
    if clean:
        df["question"] = df["question"].map(strip_prefix)
        # The same question appears under several templates; once the
        # template is gone those rows are exact duplicates.
        df = df.drop_duplicates(subset=["question", "answer"])
    return df.reset_index(drop=True)[["question", "answer"]]
