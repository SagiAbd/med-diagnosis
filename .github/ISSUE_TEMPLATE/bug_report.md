---
name: Bug Report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''
---

## 🚨 Before Creating an Issue

Please check our [Troubleshooting Guide](docs/troubleshooting.md) first. Many common issues can be resolved using the guide.

## Description
A clear and concise description of the issue.

## Steps To Reproduce
1. 
2. 
3. 

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
What actually happened.

## Troubleshooting Steps Taken
Please list the steps from the troubleshooting guide you've already tried:

- [ ] Checked container status (`docker ps -a`)
- [ ] Reviewed service logs (`docker compose logs`)
- [ ] Verified database connection
- [ ] Checked environment variables

## Environment
- OS: [e.g. Ubuntu 22.04]
- Docker version: [e.g. 24.0.5]
- Docker Compose version: [e.g. 2.21.0]

## Environment Variables
Please provide your `.env` file contents (remove any sensitive information):
```env
# Replace sensitive values with ***
DB_HOST=
DB_USER=
DB_PASSWORD=*** # Do not share actual password
DB_NAME=

# Add any other environment variables you've modified
```

## Logs
Please provide relevant logs from:
```
# Add your logs here
```

## Additional Context
Add any other context about the problem here.

## Screenshots
If applicable, add screenshots to help explain your problem.

---
⚠️ **Note**: Make sure to remove any sensitive information (passwords, tokens, etc.) before submitting your issue.