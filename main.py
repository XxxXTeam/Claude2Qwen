import os

import uvicorn

from doc import app


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
