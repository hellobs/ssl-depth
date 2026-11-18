"""发布入口：把 scripts/review_app.py 里的 FastAPI 应用挂到 PORT 上。

平台的 Python 项目默认执行 main.py，所以这里只做转发，
真正的界面代码仍在 scripts/review_app.py。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import review_app  # noqa: E402

app = review_app.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
