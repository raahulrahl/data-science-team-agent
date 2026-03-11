<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">Data Science Team Agent</h1>

<p align="center">
  <strong>LLM-based data science supervisor that generates analysis plans and recommendations for CSV datasets</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/data-science-team-agent/actions/workflows/build-and-push.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/data-science-team-agent/build-and-push.yml?branch=main" alt="Build status">
  </a>
  <a href="https://img.shields.io/github/license/Paraschamoli/data-science-team-agent">
    <img src="https://img.shields.io/github/license/Paraschamoli/data-science-team-agent" alt="License">
  </a>
</p>

---

## 📖 Overview

An intelligent agent that loads CSV datasets from URLs, inspects their structure, and generates comprehensive data science workflow plans. Built on the [Bindu Agent Framework](https://github.com/getbindu/bindu) for the Internet of Agents.

**Key Capabilities:**
- 📊 **CSV Loading**: Loads datasets from HTTP/HTTPS URLs via pandas
- 🔍 **Dataset Inspection**: Analyzes shape, columns, data types, missing values, and sample data
- 📋 **Analysis Planning**: LLM-generated workflow plans and recommendations
- 💡 **Best Practices**: Methodology guidance and data science best practices
- 🎯 **Conceptual Analysis**: Provides guidance even without datasets

**Note:** This agent provides planning and recommendations only. It does not execute actual ML models, generate visualizations, or perform statistical computations.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- API key for OpenRouter (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/data-science-team-agent.git
cd data-science-team-agent

# Create virtual environment
uv venv --python 3.12.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
```

### Configuration

Edit `.env` and add your API keys:

| Key | Get It From | Required |
|-----|-------------|----------|
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/keys) | ✅ Yes |
| `MEM0_API_KEY` | [Mem0 Dashboard](https://app.mem0.ai/dashboard/api-keys) | ❌ Optional |

### Run the Agent

```bash
# Start the agent
uv run python -m data_science_team_agent

# Agent will be available at http://localhost:3773
```

---

## 💡 Usage

### Example Queries

```bash
# Analyze a CSV dataset from URL
"Analyze this dataset: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

# Get analysis plan for specific task
"Create a workflow plan for exploratory data analysis on customer churn data"

# Request methodology guidance
"What's the best approach to handle missing values in time series data?"

# Get recommendations without dataset
"How should I approach building a classification model for customer segmentation?"
```

### Input Formats

**Plain Text:**
```
Analyze [dataset URL] and provide workflow recommendations
```

**JSON:**
```json
{
  "content": "Analyze https://example.com/data.csv",
  "focus": "exploratory-analysis"
}
```

### Output Structure

The agent returns structured output with:
- **Dataset Debug Info**: Request ID, hash, shape, columns, sample data
- **Detailed Workflow Plan**: Phase-by-phase analysis approach
- **Key Insights**: Dataset characteristics and recommendations
- **Next Steps**: Agent assignments, timeline, and deliverables

### Real Example: Tips Dataset Analysis

**Input Query:**
```
Analyze the dataset located at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
```

**Output Sample:**

```markdown
📊 Data Science Team Analysis (Supervisor Coordinated)

📊 Dataset Debug Info:
- Request ID: 2465177073088
- Dataset Hash: 15c375ad
- Shape: (244, 7)
- Columns: ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
- Sample Data:
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3

🔧 Analysis Results:

1. DETAILED WORKFLOW PLAN:

Phase 1: Data Preparation & Initial Analysis
- Load and validate dataset structure (complete)
- Confirm data quality (no missing values detected)
- Convert categorical variables (sex, smoker, day, time)
- Split data into training/testing sets

Phase 2: Exploratory Data Analysis (EDA)
- Analyze distributions of numerical variables
- Examine relationships between variables
- Create visualization suite for key insights
- Calculate summary statistics by groups

Phase 3: Feature Engineering & Modeling
- Create derived features (tip percentage, etc.)
- Prepare categorical variables for modeling
- Build and compare models (Linear Regression, Decision Tree)
- Evaluate model performance

2. KEY INSIGHTS & RECOMMENDATIONS:
[Dataset characteristics and recommended approaches]

3. NEXT STEPS FOR DATA SCIENCE TEAM:
[Agent assignments and timeline]
```

**Key Features Demonstrated:**
- ✅ Automatic CSV loading from URL
- ✅ Dataset inspection with metadata
- ✅ Comprehensive workflow planning
- ✅ Phase-by-phase recommendations
- ✅ Agent coordination suggestions

---

## 🔌 API Usage

The agent exposes a RESTful API when running. Default endpoint: `http://localhost:3773`

### Quick Start

For complete API documentation, request/response formats, and examples, visit:

📚 **[Bindu API Reference - Send Message to Agent](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)**


### Additional Resources

- 📖 [Full API Documentation](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)
- 📦 [Postman Collections](https://github.com/GetBindu/Bindu/tree/main/postman/collections)
- 🔧 [API Reference](https://docs.getbindu.com)

---

## 🎯 Skills

### data-science (v1.0.0)

**Primary Capability:**
- LLM-based supervisor coordination for data science planning
- Loads CSV datasets from URLs and generates comprehensive analysis plans
- Provides workflow recommendations without executing actual analysis

**Features:**
- CSV loading from HTTP/HTTPS URLs via pandas
- Dataset inspection (shape, columns, dtypes, missing values, sample data)
- LLM-generated analysis plans and recommendations
- Workflow planning for data science tasks
- Conceptual analysis when no dataset is provided

**Best Used For:**
- Planning data science workflows
- Getting analysis recommendations for CSV datasets
- Understanding dataset structure and characteristics
- Generating step-by-step analysis plans

**Not Suitable For:**
- Executing actual ML model training
- Generating actual visualizations or plots
- Real-time data processing
- Non-CSV data formats (JSON, Parquet, databases)
- Local file analysis (URLs only)

**Performance:**
- Average processing time: ~15 seconds
- Max concurrent requests: 5
- Memory per request: 256MB

---

## 🐳 Docker Deployment

### Local Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Agent will be available at http://localhost:3773
```

### Docker Configuration

The agent runs on port `3773` and requires:
- `OPENROUTER_API_KEY` environment variable
- `MEM0_API_KEY` environment variable (optional)

Configure these in your `.env` file before running.

### Production Deployment

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🌐 Deploy to bindus.directory

Make your agent discoverable worldwide and enable agent-to-agent collaboration.

### Setup GitHub Secrets

```bash
# Authenticate with GitHub
gh auth login

# Set deployment secrets
gh secret set BINDU_API_TOKEN --body "<your-bindu-api-key>"
gh secret set DOCKERHUB_TOKEN --body "<your-dockerhub-token>"
```

Get your keys:
- **Bindu API Key**: [bindus.directory](https://bindus.directory) dashboard
- **Docker Hub Token**: [Docker Hub Security Settings](https://hub.docker.com/settings/security)

### Deploy

```bash
# Push to trigger automatic deployment
git push origin main
```

GitHub Actions will automatically:
1. Build your agent
2. Create Docker container
3. Push to Docker Hub
4. Register on bindus.directory

---

## 🛠️ Development

### Project Structure

```
data-science-team-agent/
├── data_science_team_agent/
│   ├── skills/
│   │   └── data-science/
│   │       ├── skill.yaml          # Skill configuration
│   │       └── __init__.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py                     # Agent entry point
│   └── agent_config.json           # Agent configuration
├── tests/
│   └── test_main.py
├── .env.example
├── docker-compose.yml
├── Dockerfile.agent
└── pyproject.toml
```

### Running Tests

```bash
make test              # Run all tests
make test-cov          # With coverage report
```

### Code Quality

```bash
make format            # Format code with ruff
make lint              # Run linters
make check             # Format + lint + test
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run -a
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Powered by Bindu

Built with the [Bindu Agent Framework](https://github.com/getbindu/bindu)

**Why Bindu?**
- 🌐 **Internet of Agents**: A2A, AP2, X402 protocols for agent collaboration
- ⚡ **Zero-config setup**: From idea to production in minutes
- 🛠️ **Production-ready**: Built-in deployment, monitoring, and scaling

**Build Your Own Agent:**
```bash
uvx cookiecutter https://github.com/getbindu/create-bindu-agent.git
```

---

## 📚 Resources

- 📖 [Full Documentation](https://Paraschamoli.github.io/data-science-team-agent/)
- 💻 [GitHub Repository](https://github.com/Paraschamoli/data-science-team-agent/)
- 🐛 [Report Issues](https://github.com/Paraschamoli/data-science-team-agent/issues)
- 💬 [Join Discord](https://discord.gg/3w5zuYUuwt)
- 🌐 [Agent Directory](https://bindus.directory)
- 📚 [Bindu Documentation](https://docs.getbindu.com)

---

<p align="center">
  <strong>Built with 💛 by the team from Amsterdam 🌷</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/data-science-team-agent">⭐ Star this repo</a> •
  <a href="https://discord.gg/3w5zuYUuwt">💬 Join Discord</a> •
  <a href="https://bindus.directory">🌐 Agent Directory</a>
</p>
