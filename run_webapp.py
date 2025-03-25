import os
import sys

# If this file is executed directly, the Flask app will be imported and started
if __name__ == "__main__":
    from webapp.app import app
    
    # Run the Flask app on port 5002
    app.run(debug=True, port=5002) 