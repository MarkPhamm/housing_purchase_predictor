# 🚀 CI/CD Pipeline Guide

## 📋 Overview

This project includes a comprehensive CI/CD pipeline with GitHub Actions that automatically:

- ✅ **Tests** your code quality and functionality
- 🤖 **Trains** machine learning models
- 🌐 **Deploys** the Streamlit app
- 🔒 **Scans** for security vulnerabilities
- 📊 **Monitors** performance
- 🏷️ **Creates** releases

## 🔄 Workflows

### 1. **CI Pipeline** (`.github/workflows/ci.yml`)

**Triggers:** Push to main/develop, Pull Requests

**Jobs:**

- 🧪 **Test & Code Quality**: Unit tests, linting, formatting
- 🤖 **Train & Validate Model**: Model training and validation
- 🌐 **Test Streamlit App**: App functionality testing
- 🔒 **Security Scan**: Vulnerability scanning
- 📚 **Documentation Check**: Doc completeness verification
- ⚡ **Performance Test**: Speed and efficiency testing

### 2. **Deploy Pipeline** (`.github/workflows/deploy.yml`)

**Triggers:** Push to main, Manual trigger

**Jobs:**

- 🌐 **Deploy to Streamlit Cloud**: Automatic app deployment
- 🐳 **Build & Push Docker Image**: Container deployment
- 🚀 **Deploy to Heroku**: Cloud platform deployment
- 📢 **Notify Deployment**: Status notifications

### 3. **Release Pipeline** (`.github/workflows/release.yml`)

**Triggers:** Version tags (v1.0.0), Manual trigger

**Jobs:**

- 🏷️ **Create GitHub Release**: Automated release creation
- 📦 **Package Release**: Zip file with all assets
- 📝 **Generate Release Notes**: Comprehensive changelog

### 4. **Scheduled Maintenance** (`.github/workflows/scheduled.yml`)

**Triggers:** Every Monday 9 AM UTC, Manual trigger

**Jobs:**

- 🤖 **Weekly Model Retraining**: Keep models fresh
- 🔒 **Security Check**: Dependency vulnerability scanning
- 📚 **Documentation Check**: Doc quality monitoring
- ⚡ **Performance Check**: Speed benchmarking
- 📊 **Weekly Report**: Comprehensive status report

### 5. **Docker Pipeline** (`.github/workflows/docker.yml`)

**Triggers:** Push to main/develop, Pull Requests

**Jobs:**

- 🐳 **Build & Test Docker Image**: Container testing

## 🚀 Quick Start

### 1. **Enable GitHub Actions**

1. Push your code to GitHub
2. Go to your repository → Actions tab
3. Enable GitHub Actions if prompted

### 2. **Set Up Secrets** (Optional)

For deployment features, add these secrets in GitHub Settings → Secrets:

```
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-password
HEROKU_API_KEY=your-heroku-api-key
HEROKU_APP_NAME=your-heroku-app-name
HEROKU_EMAIL=your-email@example.com
```

### 3. **Test the Pipeline**

```bash
# Push to trigger CI
git add .
git commit -m "Add CI/CD pipeline"
git push origin main

# Check status
# Go to GitHub → Actions tab
```

## 📊 Pipeline Status

### ✅ **All Green** - Everything working

- All tests pass
- Model trains successfully
- App deploys without issues
- No security vulnerabilities

### ⚠️ **Yellow/Warning** - Minor issues

- Some non-critical tests fail
- Performance slightly degraded
- Documentation needs updates

### ❌ **Red/Failed** - Critical issues

- Core functionality broken
- Model training fails
- App won't start
- Security vulnerabilities found

## 🔧 Troubleshooting

### **Common Issues:**

#### 1. **Tests Failing**

```bash
# Run tests locally first
python -m pytest tests/ -v

# Check specific test
python -m pytest tests/test.py::TestDataPreprocessing::test_engineer_features -v
```

#### 2. **Model Training Fails**

```bash
# Check data file exists
ls -la data/global_house_purchase_dataset.csv

# Run training locally
cd src
python main_pipeline.py
```

#### 3. **App Won't Start**

```bash
# Test app locally
streamlit run app/streamlit_app.py

# Check for syntax errors
python -m py_compile app/streamlit_app.py
```

#### 4. **Docker Build Fails**

```bash
# Build locally
docker build -t house-purchase-predictor .

# Test locally
docker run -p 8501:8501 house-purchase-predictor
```

## 📈 Monitoring

### **GitHub Actions Dashboard**

- Go to your repo → Actions tab
- View all workflow runs
- Check individual job logs
- Download artifacts

### **Key Metrics to Watch:**

- **Build Time**: Should be < 10 minutes
- **Test Coverage**: Should be > 80%
- **Model Accuracy**: Should be > 95%
- **App Startup Time**: Should be < 30 seconds

## 🎯 Best Practices

### **Before Pushing:**

1. ✅ Run tests locally: `python -m pytest tests/`
2. ✅ Check code formatting: `black --check src/ tests/ app/`
3. ✅ Verify app works: `streamlit run app/streamlit_app.py`
4. ✅ Update documentation if needed

### **Commit Messages:**

```bash
# Good examples
git commit -m "feat: add new prediction feature"
git commit -m "fix: resolve model loading issue"
git commit -m "docs: update README with new instructions"

# Bad examples
git commit -m "fix stuff"
git commit -m "updates"
```

### **Branch Strategy:**

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `hotfix/*`: Critical fixes

## 🔄 Workflow Triggers

### **Automatic Triggers:**

- **Push to main**: Full CI + Deploy
- **Push to develop**: Full CI only
- **Pull Request**: Full CI only
- **Version tag**: Release creation
- **Weekly schedule**: Maintenance tasks

### **Manual Triggers:**

- **workflow_dispatch**: Run any workflow manually
- **GitHub UI**: Actions tab → Run workflow

## 📊 Artifacts

### **Generated Artifacts:**

- `trained-models`: Model files from training
- `deployment-package`: Complete app package
- `coverage-report`: Test coverage HTML
- `safety-report`: Security vulnerability report

### **Download Artifacts:**

1. Go to Actions tab
2. Click on a workflow run
3. Scroll to "Artifacts" section
4. Download the files you need

## 🚀 Deployment Options

### **1. Streamlit Cloud** (Recommended)

- Free hosting
- Automatic deployments
- Custom domains
- Easy setup

### **2. Docker Hub**

- Container registry
- Easy deployment anywhere
- Version management
- Public/private repos

### **3. Heroku**

- Cloud platform
- Easy scaling
- Add-ons available
- Professional hosting

### **4. Self-Hosted**

- Full control
- Custom infrastructure
- Cost-effective
- More complex setup

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Docker Documentation](https://docs.docker.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)

## 🆘 Support

If you encounter issues with the CI/CD pipeline:

1. **Check the logs** in GitHub Actions
2. **Run locally** to reproduce the issue
3. **Check this guide** for common solutions
4. **Create an issue** with detailed error information
5. **Check the troubleshooting section** in the main README

---

**Happy Deploying! 🚀**
