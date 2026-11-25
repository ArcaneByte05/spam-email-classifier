# Spam Email Classifier

This internship project reads an English email's subject and body and predicts **spam** or **not spam**. In this dataset, “ham” means an ordinary, non-spam email. You can paste a message into a local web app and see its prediction.

## Objectives and tools

- Read labelled email content with Pandas and clean the message text with Python.
- Use scikit-learn to turn words into TF-IDF features and train a Multinomial Naive Bayes classifier.
- Measure accuracy on unseen emails and show the spam/ham distribution with Streamlit.

## Files and data

| File | Purpose |
| --- | --- |
| `app.py` | Collects email content, predicts a label and displays the chart. |
| `train.py` | Reads the dataset, cleans messages, trains the model and prints results. |
| `requirements.txt` | Lists the three Python packages to install. |
| `README.md` | Setup, code explanation, measured results and 10 viva questions. |
| `enron_spam_data.zip` | Download once in step 5. It contains a single CSV with 33,716 emails. |

**Dataset credit.** The Enron-Spam corpus was prepared by V. Metsis, I. Androutsopoulos and G. Paliouras. This project reads M. Wiechmann's CSV version. The original records are email messages with subjects and bodies. [Dataset, column descriptions and original research credit](https://github.com/MWiechmann/enron_spam_data)

## Run on Windows 11 with VS Code

1. Install 64-bit Python 3.12. The [Python 3.12.10 release page](https://www.python.org/downloads/release/python-31210/) has a **Windows installer (64-bit)** under Files. In the installer, select **Add python.exe to PATH** and enable the Python launcher if offered. Select **Install Now**.
2. Install [VS Code for Windows x64](https://code.visualstudio.com/download). Open it, press **Ctrl+Shift+X**, search for **Python**, and install the extension published by Microsoft.
3. In File Explorer, create an **Internship Projects** folder inside Documents. Right-click the project ZIP, select **Extract All**, and choose your new folder as the destination. In VS Code select **File > Open Folder** and open the extracted project folder that directly contains both Python files. If you are building by hand, create a folder with the repository name and copy the supplied files into it using **File > New Text File** and **Save As**.
4. Select **Terminal > New Terminal**. Use PowerShell. Run the commands below one line at a time. The first command should print Python 3.12.x. The environment keeps this project's packages together. Calling its Python directly also works when PowerShell script activation is disabled.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Download the dataset directly into the open project folder. Keep this **dataset ZIP** compressed. Pandas opens the CSV inside it.

```powershell
curl.exe -fL "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip" -o enron_spam_data.zip
```

If you prefer the browser, [download the same dataset](https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip) and move it next to `train.py`. Check that its name is exactly `enron_spam_data.zip`.

6. Train and test the model. Look for the test accuracy and the table for each class in the terminal.

```powershell
.\.venv\Scripts\python.exe train.py
```

7. Launch the app in the same terminal.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

8. Open the **Local URL** printed in the terminal (usually http://localhost:8501). Type your example in the box and click the prediction button. Scroll down to see the accuracy and the distribution graph. Leave the terminal open while using the app. Press **Ctrl+C** in the terminal to stop it. On later visits, repeat step 7.

The terminal training run and the app are two entry points to the same function. The app trains once when it starts, holds the model in memory, and reuses it between clicks. You can launch the app directly after setup. A saved model file is optional for this design.

**If a command fails**

- If Windows cannot find `py`, reopen VS Code after installing Python. If `python --version` already prints 3.12.x, use `python -m venv .venv` in step 4.
- If Python cannot find `requirements.txt` or `train.py`, run `dir` in the terminal. Open the folder that contains those files.
- If VS Code underlines an import after installation, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select the project's `.venv`.

## Follow the code

1. **Read the dataset.** In `train.py`, `load_emails()` reads the subject, message and label. It joins subject and body with a space so an email can still be used if one field is empty. Message ID, date and the answer column are excluded from the input to the model.
2. **Clean the text.** `clean_email()` converts to lowercase, removes HTML tags, links and email addresses, and extracts words of at least two characters. Empty examples and conflicting labels for the same cleaned text are removed. Repeated copies of the same cleaned email count once.
3. **Keep a test set.** `train_test_split` reserves 20% of the messages for testing. Stratification keeps spam and ham in similar proportions across the two sets. Seed 42 repeats this split for the same data in the same order.
4. **Create TF-IDF features.** Words present in many messages tend to carry less information. TF-IDF represents an email as word weights. The vectorizer learns its vocabulary only from the training messages. It ignores words that appear in only one training email and keeps up to 20,000 terms to limit memory use.
5. **Train Naive Bayes.** `MultinomialNB` learns how strongly each word is associated with spam or ham. It combines evidence from the words present with the overall frequency of each class. Its “naive” assumption is that words give independent evidence once the class is known. The `alpha=1.0` setting adds smoothing so a word absent from one class does not wipe out that class's score.
6. **Test.** The pipeline predicts labels for the held-out messages. The code prints accuracy, a per-class report and a small table called a confusion matrix. Rows are actual labels and columns are predictions.
7. **Use Streamlit.** In `app.py`, the button calls the same cleaner and fitted pipeline for your email. If no words are recognized, the app asks for more text. The graph counts actual spam and ham labels after cleaning.

## Measured result and demo

After cleaning, **30,056** emails remained: 15,527 ham and 14,529 spam. Training used **24,044**. The model correctly classified **5,936 of 6,012** test emails (**98.74% accuracy**).

| Actual label | Predicted ham | Predicted spam |
| --- | ---: | ---: |
| Ham | 3,068 | 38 |
| Spam | 38 | 2,868 |

For example, the top-right cell means 38 legitimate emails were flagged as spam. These results come from a random holdout of this historical corpus. Messages from the same thread or with similar wording may occur in both sets, and today's inbox may contain very different spam.

| Try this email | Observed prediction |
| --- | --- |
| Free cash prize! Click here to claim your reward and win money now. | spam |
| Team meeting tomorrow at 10. Please review the attached agenda before the meeting. | not spam (ham) |

## 10 viva questions

1. **What does the classifier predict?** It reads an email and chooses between spam and ham.
2. **What is ham?** It is the dataset's label for a legitimate email.
3. **Which features did you use?** The model uses TF-IDF weights from the subject and body. The spam/ham label is the target.
4. **Why clean email text?** Tags, addresses and differences in case can create many versions of similar content. Cleaning gives the model a more consistent input.
5. **What does TF-IDF do?** It converts text to numbers. It gives weight to terms present in an email and discounts terms found across many emails.
6. **How does Naive Bayes work?** It learns which words tend to occur in each class and combines evidence from the words in a new message to pick a label.
7. **What is “naive” about it?** It treats the words as independent pieces of evidence given the class, even though words in real sentences are related.
8. **What is smoothing?** It adds a small amount to the learned word counts. A word never seen in one class can then receive a small probability for that class.
9. **What is a false positive here?** A legitimate email marked as spam. There were 38 in the held-out test set.
10. **How did you measure accuracy?** Divide correct predictions by all test emails: 5,936 divided by 6,012 gives 98.74%. The emails used for this calculation were held out during training.

## Publish on GitHub

1. Sign in to [GitHub](https://github.com/) or create a free account. Open [New repository](https://github.com/new).
2. Enter **spam-email-classifier**, choose **Public** so your reviewer can open it, leave the initialization options unchecked, and click **Create repository**.
3. On the empty repository page, click **uploading an existing file**. If the repository already has a file, choose **Add file > Upload files**.
4. Open the extracted folder in File Explorer. Select only **app.py, train.py, requirements.txt and README.md** (the README links the dataset) and drag them into the browser together. This puts the README and source code at the repository root.
5. Enter a message such as **Add working classifier and README**, then click **Commit changes**. Check that GitHub displays the README below the file list. Copy the repository URL from your browser.
6. Repeat for the other internship project using its own repository name. Submit both repository links by email with your offer letter attached privately, as requested by Codec Technologies.

[GitHub's file upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
