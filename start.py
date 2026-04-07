import os
import uvicorn

port = int(os.environ.get("PORT", 8001))
uvicorn.run("api:app", host="0.0.0.0", port=port)
