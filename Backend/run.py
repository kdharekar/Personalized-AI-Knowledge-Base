import uvicorn
import os
import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    print ("Running Uvicorn server...")
    uvicorn.run(
        "app.main:app",   # module:object
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        reload_dirs=["app"],
        log_level="debug"
    )
    print ("Server stopped.")
