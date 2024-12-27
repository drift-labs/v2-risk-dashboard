
# Drift v2 Risk Dashboard Codebase Analysis

## Directory Tree

```
📄 .dockerignore
📄 .env.example
📂 .github
    📂 workflows
        📄 master.yaml
📄 .gitignore
📄 .pre-commit-config.yaml
📂 .streamlit
    📄 config.toml
📄 Dockerfile-backend
📄 Dockerfile-frontend
📄 README.md
📂 backend
    📂 api
        📄 __init__.py
        📄 asset_liability.py
        📄 health.py
        📄 liquidation.py
        📄 metadata.py
        📄 price_shock.py
        📄 snapshot.py
        📄 ucache.py
    📄 app.py
    📂 middleware
        📄 cache_middleware.py
        📄 readiness.py
    📂 scripts
        📄 generate_ucache.py
    📄 state.py
    📂 utils
        📄 matrix.py
        📄 repeat_every.py
        📄 user_metrics.py
        📄 vat.py
        📄 waiting_for.py
📄 gunicorn_config.py
📂 images
    📄 drift.svg
    📄 driftlogo.png
📄 requirements.txt
📂 src
    📂 lib
        📄 api.py
        📄 page.py
        📄 user_metrics.py
    📄 main.py
    📂 page
        📄 asset_liability.py
        📄 backend.py
        📄 health.py
        📄 health_cached.py
        📄 liquidation_curves.py
        📄 orderbook.py
        📄 price_shock.py
        📄 welcome.py
    📄 style.css
    📄 utils.py

```

---


## Consolidated Analysis of the Application

### Overview of the Application
- **Purpose**: 
  The application monitors and visualizes financial metrics such as leverage, asset-liability matrices, and risk scenarios for a decentralized finance (DeFi) ecosystem. 
- **Technologies**:
  - **Back-End**: FastAPI, Driftpy, Python
  - **Front-End**: Streamlit, Plotly, Pandas
  - **Database/Storage**: Caching via pickled files
  - **Others**: Docker, Gunicorn, `.env` configuration

### Back-End Analysis
- **Architecture**:
  - Built with FastAPI, the back-end provides modular APIs organized into directories like `asset_liability`, `price_shock`, and `health`.
  - Centralized state management via `BackendState` class, integrating Solana blockchain data with `Driftpy`.
- **Data Flow**:
  - Data is fetched from the Solana blockchain using `AsyncClient` and cached as pickled files for performance.
  - The `load_pickle_snapshot` function loads the latest cached data into a shared `Vat` object.
  - API endpoints expose this data for front-end consumption.
- **Integrations**:
  - **Driftpy**: Handles interactions with Solana blockchain and the Drift protocol.
  - **Solana**: Provides oracle data, user accounts, and market metrics.
- **Key Services/Modules**:
  - `asset_liability`: Computes asset-liability matrices.
  - `price_shock`: Analyzes leverage and bankruptcy risk under oracle price distortions.
  - `health`: Provides account and system health metrics.

### Front-End Analysis
- **Framework**: Streamlit
- **Structure**:
  - **Main Entry**: `main.py` sets up the application with configuration, styling, and navigation.
  - **Pages**: Modular design with individual pages (`asset_liability.py`, `price_shock.py`) for specific metrics.
  - **Styling**: Custom CSS for enhanced UI, applied via `style.css`.
- **UI/UX Features**:
  - Interactive data visualizations powered by Plotly.
  - Sidebar navigation for seamless exploration of metrics.
  - Clear, concise presentation of complex data using Pandas for preprocessing.

### Component-Level Analysis
#### Back-End
- **`BackendState`**:
  - Manages connections to Solana and Driftpy clients.
  - Caches data for efficient access and uses the `Vat` object to process metrics.
- **APIs**:
  - `/matrix`: Fetches asset-liability matrix data.
  - `/price_shock`: Provides leverage and bankruptcy metrics under varying conditions.

#### Front-End
- **`asset_liability.py`**:
  - Fetches matrix data and presents it in a summarized form.
  - Highlights financial risks using Pandas transformations.
- **`price_shock.py`**:
  - Models the effect of price changes on user leverage and bankruptcies.
  - Visualizes results dynamically with Plotly.

### Application Logic
- **Core Processes**:
  - Data is fetched, cached, and preprocessed in the back-end.
  - The front-end interacts with APIs to retrieve and visualize metrics.
- **Inter-Component Communication**:
  - Front-end fetches data dynamically using modular API calls (`lib.api.api2`).
  - Back-end orchestrates data processing via `BackendState` and utility modules.

### Development Practices
- **Strengths**:
  - Modular design: Clear separation of concerns across back-end and front-end.
  - Code reusability: Utility functions like `generate_summary_data` and `load_newest_files` are well-structured.
  - Environment flexibility: `.env` variables ensure easy deployment in different environments.
- **Potential Improvements**:
  - Enhance documentation for API endpoints and utility functions.
  - Add error handling for API calls to improve user experience during network issues.

### Key Features
- **Back-End**:
  - Efficient caching mechanism for state persistence and fast restarts.
  - Real-time interaction with Solana blockchain data.
- **Front-End**:
  - Interactive visualizations for exploring financial risks.
  - Modular and extensible design to accommodate new metrics.

### Deployment and Configuration
- **Deployment Instructions**:
  - Back-End: Run Gunicorn with `gunicorn backend.app:app -c gunicorn_config.py`.
  - Front-End: Launch with `streamlit run src/main.py`.
- **Environment Configurations**:
  - Uses `.env` file for critical variables like `BACKEND_URL` and `RPC_URL`.
  - Dockerized setup with separate Dockerfiles for front-end and back-end.
- **CI/CD**:
  - GitHub workflows might automate deployment (files in `.github` directory).

### Summary of Findings
- **Strengths**:
  - The application is modular, scalable, and designed for analyzing complex financial metrics.
  - Interactive front-end enhances usability and understanding of data.
- **Areas for Improvement**:
  - Improve inline documentation and logging for better maintainability.
  - Implement robust error handling for API integrations.
  - Consider database integration for historical data analysis instead of relying solely on pickled files.

