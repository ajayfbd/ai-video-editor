# Documentation Quality Assurance Scripts

This directory contains automated tools for maintaining documentation quality and preventing redundancy in the AI Video Editor project.

## Scripts Overview

### `validate_documentation.py`
Comprehensive documentation validation script that checks:
- Internal link integrity
- Heading structure and hierarchy
- Code example syntax
- Content quality standards
- Cross-reference accuracy

**Usage:**
```bash
# Basic validation (warnings and errors)
python scripts/validate_documentation.py

# Only show errors
python scripts/validate_documentation.py --severity error

# Generate JSON report
python scripts/validate_documentation.py --format json --output validation-report.json

# Validate specific directory
python scripts/validate_documentation.py --root docs/user-guide/
```

### `run_maintenance_checks.py`
Automated maintenance system that runs scheduled checks based on configuration:

**Usage:**
```bash
# Daily checks (critical issues only)
python scripts/run_maintenance_checks.py daily

# Weekly checks (warnings and errors)
python scripts/run_maintenance_checks.py weekly

# Monthly comprehensive review
python scripts/run_maintenance_checks.py monthly

# Quarterly structural analysis
python scripts/run_maintenance_checks.py quarterly
```

### `documentation_config.json`
Configuration file that defines:
- Validation settings and severity levels
- Documentation structure and ownership
- Quality standards and requirements
- Maintenance schedules and procedures

## Quick Start

### 1. Run Initial Validation
```bash
# Check for critical issues
python scripts/validate_documentation.py --severity error

# If no errors, run full validation
python scripts/validate_documentation.py
```

### 2. Set Up Regular Maintenance
```bash
# Add to crontab for daily checks
0 9 * * * cd /path/to/project && python scripts/run_maintenance_checks.py daily

# Weekly checks every Monday
0 9 * * 1 cd /path/to/project && python scripts/run_maintenance_checks.py weekly

# Monthly checks on first day of month
0 9 1 * * cd /path/to/project && python scripts/run_maintenance_checks.py monthly
```

### 3. Pre-commit Hook (Optional)
```bash
# Add validation to pre-commit hook
echo "python scripts/validate_documentation.py --severity error" >> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Documentation Quality
on: [push, pull_request]
jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Validate Documentation
        run: python scripts/validate_documentation.py --severity error
```

## Validation Rules

### Link Validation
- ✅ Internal links must point to existing files
- ✅ Anchor links must reference existing headings
- ✅ Relative paths must be correct
- ⚠️ External links are not validated by default

### Content Quality
- ✅ Files must meet minimum content length
- ⚠️ TODO/FIXME comments are flagged for review
- ✅ Code blocks are checked for basic syntax issues
- ✅ Heading hierarchy must be logical (no level jumps)

### Structure Standards
- ✅ Consistent formatting across all files
- ✅ Required sections present in appropriate documents
- ✅ Cross-references are functional and relevant
- ✅ Navigation paths are complete

## Maintenance Schedule

### Daily (Automated)
- Link integrity validation
- Critical error detection
- File accessibility checks

### Weekly (Automated)
- Full validation with warnings
- Content quality review
- Cross-reference audit

### Monthly (Semi-automated)
- Content freshness analysis
- User journey validation
- Comprehensive quality review

### Quarterly (Manual + Automated)
- Structural analysis
- Documentation strategy review
- Process improvement assessment

## Troubleshooting

### Common Issues

**"Broken internal link" errors:**
- Check if the target file exists
- Verify the relative path is correct
- Ensure anchor references match actual headings

**"Heading hierarchy" warnings:**
- Don't skip heading levels (e.g., H1 → H3)
- Use consistent heading structure
- Consider document organization

**"Content length" warnings:**
- Files under 100 characters may be incomplete
- Consider if the file serves a real purpose
- Add meaningful content or remove the file

### Getting Help

1. Check the validation output for specific error details
2. Review the maintenance guidelines in `docs/support/documentation-maintenance.md`
3. Run validation with `--format json` for detailed analysis
4. Check the configuration in `documentation_config.json`

## Configuration

### Customizing Validation Rules
Edit `documentation_config.json` to:
- Adjust severity levels for different issue types
- Modify file inclusion/exclusion patterns
- Update quality standards and requirements
- Change maintenance schedules

### Adding New Validation Rules
Extend `validate_documentation.py` by:
1. Adding new validation methods to `DocumentationValidator`
2. Updating the configuration schema
3. Adding appropriate error handling
4. Testing with sample documentation

## Reports

### Report Locations
- Daily/Weekly reports: Console output
- Monthly reports: `reports/monthly-maintenance-YYYY-MM.md`
- Quarterly reports: `reports/quarterly-maintenance-YYYY-QX.md`
- JSON reports: Specified output file or stdout

### Report Contents
- Issue summary by type and severity
- File-by-file breakdown of problems
- Trend analysis (for regular reports)
- Recommendations for improvement

This system ensures consistent, high-quality documentation while minimizing maintenance overhead and preventing redundancy.