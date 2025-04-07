# GetImages/__init__.py

import logging
import azure.functions as func
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Return a simple success message with empty images array
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Minimal GetImages function is running",
                "images": [],
                "container": "images1"
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error in GetImages function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )