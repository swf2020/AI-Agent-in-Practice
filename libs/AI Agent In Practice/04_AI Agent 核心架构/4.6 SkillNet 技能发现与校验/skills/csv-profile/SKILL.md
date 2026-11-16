---
name: csv-profile
description: Profile a local CSV by counting data rows and missing values.
---

# CSV profile

1. Open the user-provided CSV as UTF-8 and read its header using Python's csv module.
2. Count rows after the header and count empty fields for each column.
3. Report the row count and the missing-value counts. Keep the input file unchanged.
