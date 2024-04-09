from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "Pointcept/PointTransformerV3"
FILENAME = "s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth"
# print((hf_hub_download.__dict__))
model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)