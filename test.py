# Test MatchingModule
from ultralytics.nn.modules import MatchingModule
mm = MatchingModule()
q = torch.randn(4, 256, 40, 40)
s = torch.randn(4, 256, 40, 40)
output = mm(q, s)
assert output.shape == q.shape  # ✓ Pass

# Test SiamDetectionModel
from ultralytics.nn.tasks import SiamDetectionModel
model = SiamDetectionModel("yolo11n.yaml", nc=1)
query = torch.randn(2, 3, 640, 640)
support = torch.randn(2, 3, 640, 640)
preds = model(query, support)  # ✓ Pass

# Test SiamDataset
from ultralytics.data.dataset import SiamDataset
dataset = SiamDataset(img_path="...", data=data)
batch = dataset[0]
assert "query_img" in batch  # ✓ Pass
assert "support_img" in batch  # ✓ Pass