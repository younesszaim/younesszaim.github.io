{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "view-in-github"
   },
   "source": [
    "<a href=\"https://colab.research.google.com/github/Giffy/AI_fast.ai/blob/master/Machine%20Learning/lesson1_randomforest.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "HeQi7nEXJX9t"
   },
   "source": [
    "**Important: This notebook will only work with fastai-0.7.x. Do not try to run any fastai-1.x code from this path in the repository because it will load fastai-0.7.x**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "gsb7RkmyJX-K"
   },
   "source": [
    "# 1 Intro to Random Forests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "8EFuzwYKJX-d"
   },
   "source": [
    "## 1.1 About this course"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "NpFfOmXtJX-8"
   },
   "source": [
    "### Teaching approach"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "2WPt_fKTJX_J"
   },
   "source": [
    "This course is being taught by Jeremy Howard, and was developed by Jeremy along with Rachel Thomas. Rachel has been dealing with a life-threatening illness so will not be teaching as originally planned this year.\n",
    "\n",
    "Jeremy has worked in a number of different areas - feel free to ask about anything that he might be able to help you with at any time, even if not directly related to the current topic:\n",
    "\n",
    "- Management consultant (McKinsey; AT Kearney)\n",
    "- Self-funded startup entrepreneur (Fastmail: first consumer synchronized email; Optimal Decisions: first optimized insurance pricing)\n",
    "- VC-funded startup entrepreneur: (Kaggle; Enlitic: first deep-learning medical company)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "AbTSKOzBJX_Z"
   },
   "source": [
    "I'll be using a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the \"big picture\", and don't know what they'll need.\n",
    "\n",
    "If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [in this talk](https://vimeo.com/214233053).\n",
    "\n",
    "Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.\n",
    "\n",
    "All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some \"black boxes\" such as random forests that haven't yet been explained in detail, and then we'll dig into the lower level details later.\n",
    "\n",
    "To start, focus on what things DO, not what they ARE."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Vo-cGVcxJX_u"
   },
   "source": [
    "### Your practice"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "e0E0TfHiJX_-"
   },
   "source": [
    "People learn by:\n",
    "1. **doing** (coding and building)\n",
    "2. **explaining** what they've learned (by writing or helping others)\n",
    "\n",
    "Therefore, we suggest that you practice these skills on Kaggle by:\n",
    "1. Entering competitions (*doing*)\n",
    "2. Creating Kaggle kernels (*explaining*)\n",
    "\n",
    "It's OK if you don't get good competition ranks or any kernel votes at first - that's totally normal! Just try to keep improving every day, and you'll see the results over time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "nidTaGOMJYAS"
   },
   "source": [
    "To get better at technical writing, study the top ranked Kaggle kernels from past competitions, and read posts from well-regarded technical bloggers. Some good role models include:\n",
    "\n",
    "- [Peter Norvig](http://nbviewer.jupyter.org/url/norvig.com/ipython/ProbabilityParadox.ipynb) (more [here](http://norvig.com/ipython/))\n",
    "- [Stephen Merity](https://smerity.com/articles/2017/deepcoder_and_ai_hype.html)\n",
    "- [Julia Evans](https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture) (more [here](https://jvns.ca/blog/2014/08/12/what-happens-if-you-write-a-tcp-stack-in-python/))\n",
    "- [Julia Ferraioli](http://blog.juliaferraioli.com/2016/02/exploring-world-using-vision-twilio.html)\n",
    "- [Edwin Chen](http://blog.echen.me/2014/10/07/moving-beyond-ctr-better-recommendations-through-human-evaluation/)\n",
    "- [Slav Ivanov](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b) (fast.ai student)\n",
    "- [Brad Kenstler](https://hackernoon.com/non-artistic-style-transfer-or-how-to-draw-kanye-using-captain-picards-face-c4a50256b814) (fast.ai and USF MSAN student)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "J-A6Fj4PJYAk"
   },
   "source": [
    "### Books"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "BoU0Q8qeJYA5"
   },
   "source": [
    "The more familiarity you have with numeric programming in Python, the better. If you're looking to improve in this area, we strongly suggest Wes McKinney's [Python for Data Analysis, 2nd ed](https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1491957662/ref=asap_bc?ie=UTF8).\n",
    "\n",
    "For machine learning with Python, we recommend:\n",
    "\n",
    "- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Andreas-Mueller/dp/1449369413): From one of the scikit-learn authors, which is the main library we'll be using\n",
    "- [Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow, 2nd Edition](https://www.amazon.com/Python-Machine-Learning-scikit-learn-TensorFlow/dp/1787125939/ref=dp_ob_title_bk): New version of a very successful book. A lot of the new material however covers deep learning in Tensorflow, which isn't relevant to this course\n",
    "- [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291/ref=pd_lpo_sbs_14_t_0?_encoding=UTF8&psc=1&refRID=MBV2QMFH3EZ6B3YBY40K)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "tzXIKZlNJYBH"
   },
   "source": [
    "### Syllabus in brief"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "GdKb66Z-JYBS"
   },
   "source": [
    "Depending on time and class interests, we'll cover something like (not necessarily in this order):\n",
    "\n",
    "- Train vs test\n",
    "  - Effective validation set construction\n",
    "- Trees and ensembles\n",
    "  - Creating random forests\n",
    "  - Interpreting random forests\n",
    "- What is ML?  Why do we use it?\n",
    "  - What makes a good ML project?\n",
    "  - Structured vs unstructured data\n",
    "  - Examples of failures/mistakes\n",
    "- Feature engineering\n",
    "  - Domain specific - dates, URLs, text\n",
    "  - Embeddings / latent factors\n",
    "- Regularized models trained with SGD\n",
    "  - GLMs, Elasticnet, etc (NB: see what James covered)\n",
    "- Basic neural nets\n",
    "  - PyTorch\n",
    "  - Broadcasting, Matrix Multiplication\n",
    "  - Training loop, backpropagation\n",
    "- KNN\n",
    "- CV / bootstrap (Diabetes data set?)\n",
    "- Ethical considerations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "G7Ufj_u2JYBc"
   },
   "source": [
    "Skip:\n",
    "\n",
    "- Dimensionality reduction\n",
    "- Interactions\n",
    "- Monitoring training\n",
    "- Collaborative filtering\n",
    "- Momentum and LR annealing\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "rCsFj7OsJYBm"
   },
   "source": [
    "## 1.2 Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "q4R_oS_iqckk"
   },
   "outputs": [],
   "source": [
    "!pip install fastai==0.7.0 > null\n",
    "!pip install scikit-learn==0.20 > null\n",
    "!wget https://raw.githubusercontent.com/Giffy/Personal_dataset_repository/master/train.tar.gz\n",
    "!tar xvf train.tar.gz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "VPQlhNO0JYBy"
   },
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "aJYUth0YJYDc"
   },
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'fastai.structured'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-98cc0abc5854>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mfastai\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimports\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mfastai\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstructured\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpandas_summary\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mDataFrameSummary\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mensemble\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mRandomForestRegressor\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mRandomForestClassifier\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'fastai.structured'"
     ]
    }
   ],
   "source": [
    "from fastai.imports import *\n",
    "from fastai.structured import *\n",
    "\n",
    "from pandas_summary import DataFrameSummary\n",
    "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
    "from IPython.display import display\n",
    "\n",
    "from sklearn import metrics\n",
    "set_plot_sizes(12,14,16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "abJOtBgj4ila"
   },
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "rjJTTgPeJYEf"
   },
   "outputs": [],
   "source": [
    "PATH = \"/content/data/bulldozers/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "colab_type": "code",
    "id": "0SU_NuH3JYFe",
    "outputId": "6a680969-57dd-45a8-8f98-9725742c3a66"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train.csv  training_dataset\n"
     ]
    }
   ],
   "source": [
    "!ls {PATH}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "V0TCusfWJYGi"
   },
   "source": [
    "# 2 Introduction to *Blue Book for Bulldozers*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "afyeBraRJYGz"
   },
   "source": [
    "## 2.1 About..."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "fsr6opRCJYG-"
   },
   "source": [
    "### ...our teaching"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "r2wrrTR9JYHJ"
   },
   "source": [
    "At fast.ai we have a distinctive [teaching philosophy](http://www.fast.ai/2016/10/08/teaching-philosophy/) of [\"the whole game\"](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719/ref=sr_1_1?ie=UTF8&qid=1505094653).  This is different from how most traditional math & technical courses are taught, where you have to learn all the individual elements before you can combine them (Harvard professor David Perkins call this *elementitis*), but it is similar to how topics like *driving* and *baseball* are taught.  That is, you can start driving without [knowing how an internal combustion engine works](https://medium.com/towards-data-science/thoughts-after-taking-the-deeplearning-ai-courses-8568f132153), and children begin playing baseball before they learn all the formal rules."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "nzbt-BNLJYHU"
   },
   "source": [
    "### ...our approach to machine learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "UJcxm33wJYHl"
   },
   "source": [
    "Most machine learning courses will throw at you dozens of different algorithms, with a brief technical description of the math behind them, and maybe a toy example. You're left confused by the enormous range of techniques shown and have little practical understanding of how to apply them.\n",
    "\n",
    "The good news is that modern machine learning can be distilled down to a couple of key techniques that are of very wide applicability. Recent studies have shown that the vast majority of datasets can be best modeled with just two methods:\n",
    "\n",
    "- *Ensembles of decision trees* (i.e. Random Forests and Gradient Boosting Machines), mainly for structured data (such as you might find in a database table at most companies)\n",
    "- *Multi-layered neural networks learnt with SGD* (i.e. shallow and/or deep learning), mainly for unstructured data (such as audio, vision, and natural language)\n",
    "\n",
    "In this course we'll be doing a deep dive into random forests, and simple models learnt with SGD. You'll be learning about gradient boosting and deep learning in part 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "tDEjcM07JYH2"
   },
   "source": [
    "### ...this dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "O8MYcx1jJYID"
   },
   "source": [
    "We will be looking at the Blue Book for Bulldozers Kaggle Competition: \"The goal of the contest is to predict the sale price of a particular piece of heavy equiment at auction based on it's usage, equipment type, and configuration.  The data is sourced from auction result postings and includes information on usage and equipment configurations.\"\n",
    "\n",
    "This is a very common type of dataset and prediciton problem, and similar to what you may see in your project or workplace."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "4xrtJBrbJYIO"
   },
   "source": [
    "### ...Kaggle Competitions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "chcHgLY2JYIX"
   },
   "source": [
    "Kaggle is an awesome resource for aspiring data scientists or anyone looking to improve their machine learning skills.  There is nothing like being able to get hands-on practice and receiving real-time feedback to help you improve your skills.\n",
    "\n",
    "Kaggle provides:\n",
    "\n",
    "1. Interesting data sets\n",
    "2. Feedback on how you're doing\n",
    "3. A leader board to see what's good, what's possible, and what's state-of-art.\n",
    "4. Blog posts by winning contestants share useful tips and techniques."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "0HC1kO4HJYIi"
   },
   "source": [
    "## 2.2 The data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "MXWHp01OJYIx"
   },
   "source": [
    "### 2.2.1 Look at the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "YaxqrOxnJYI8"
   },
   "source": [
    "Kaggle provides info about some of the fields of our dataset; on the [Kaggle Data info](https://www.kaggle.com/c/bluebook-for-bulldozers/data) page they say the following:\n",
    "\n",
    "For this competition, you are predicting the sale price of bulldozers sold at auctions. The data for this competition is split into three parts:\n",
    "\n",
    "- **Train.csv** is the training set, which contains data through the end of 2011.\n",
    "- **Valid.csv** is the validation set, which contains data from January 1, 2012 - April 30, 2012. You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.\n",
    "- **Test.csv** is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.\n",
    "\n",
    "The key fields are in train.csv are:\n",
    "\n",
    "- SalesID: the unique identifier of the sale\n",
    "- MachineID: the unique identifier of a machine.  A machine can be sold multiple times\n",
    "- saleprice: what the machine sold for at auction (only provided in train.csv)\n",
    "- saledate: the date of the sale"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "GS3BxRIGJYJL"
   },
   "source": [
    "*Question*\n",
    "\n",
    "What stands out to you from the above description?  What needs to be true of our training and validation sets?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ehc28DeuJYJZ"
   },
   "outputs": [],
   "source": [
    "df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, \n",
    "                     parse_dates=[\"saledate\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "LzaI4ybCJYJ5"
   },
   "source": [
    "In any sort of data science work, it's **important to look at your data**, to make sure you understand the format, how it's stored, what type of values it holds, etc. Even if you've read descriptions about your data, the actual data may not be what you expect."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "B76Sz33yJYKC"
   },
   "outputs": [],
   "source": [
    "def display_all(df):\n",
    "    with pd.option_context(\"display.max_rows\", 1000, \"display.max_columns\", 1000): \n",
    "        display(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "X9ZiKg5uJYK5"
   },
   "outputs": [],
   "source": [
    "display_all(df_raw.tail().T)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "jvIBekCwJYL0"
   },
   "outputs": [],
   "source": [
    "display_all(df_raw.describe(include='all').T)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "qEwfVhy6JYM4"
   },
   "source": [
    "It's important to note what metric is being used for a project. Generally, selecting the metric(s) is an important part of the project setup. However, in this case Kaggle tells us what metric to use: RMSLE (root mean squared log error) between the actual and predicted auction prices. Therefore we take the log of the prices, so that RMSE will give us what we need."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ugVaS8pIJYND"
   },
   "outputs": [],
   "source": [
    "df_raw.SalePrice = np.log(df_raw.SalePrice)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "SUGOtVXkJYNq"
   },
   "source": [
    "### 2.2.2 Initial processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "KJhAsczBJYN2"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_jobs=-1)\n",
    "# The following code is supposed to fail due to string values in the input data\n",
    "m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "SlXJK-7ZJYPA"
   },
   "source": [
    "This dataset contains a mix of **continuous** and **categorical** variables.\n",
    "\n",
    "The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals.  You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "txN1bQSgJYPK"
   },
   "outputs": [],
   "source": [
    "add_datepart(df_raw, 'saledate')\n",
    "df_raw.saleYear.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "cPVS2tVnJYQL"
   },
   "source": [
    "The categorical variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we call `train_cats` to convert strings to pandas categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "vXl6UN85JYQc"
   },
   "outputs": [],
   "source": [
    "train_cats(df_raw)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "VNovvwvIJYTa"
   },
   "source": [
    "We can specify the order to use for categorical variables if we wish:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "mNoKfkX5JYTx"
   },
   "outputs": [],
   "source": [
    "df_raw.UsageBand.cat.categories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "9xngR4RAJYUV"
   },
   "outputs": [],
   "source": [
    "df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "-70frqpqJYUm"
   },
   "source": [
    "Normally, pandas will continue displaying the text categories, while treating them as numerical data internally. Optionally, we can replace the text categories with numbers, which will make this variable non-categorical, like so:."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "kiHJK4AkJYUr"
   },
   "outputs": [],
   "source": [
    "df_raw.UsageBand = df_raw.UsageBand.cat.codes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "tp4d_WQVJYU-"
   },
   "source": [
    "We're still not quite done - for instance we have lots of missing values, which we can't pass directly to a random forest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "1CYmpNtOJYVG"
   },
   "outputs": [],
   "source": [
    "display_all(df_raw.isnull().sum().sort_index()/len(df_raw))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "i7HN0ijVJYVj"
   },
   "source": [
    "But let's save this file for now, since it's already in format can we be stored and accessed efficiently."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "mnbe5kXuJYVp"
   },
   "outputs": [],
   "source": [
    "os.makedirs('tmp', exist_ok=True)\n",
    "df_raw.to_feather('tmp/bulldozers-raw')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "DhcxnrqRJYV6"
   },
   "source": [
    "### 2.2.3 Pre-processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Yu1touGPJYWB"
   },
   "source": [
    "In the future we can simply read it from this fast format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "E3L0XRJTJYWH"
   },
   "outputs": [],
   "source": [
    "import feather\n",
    "df_raw=feather.read_dataframe('tmp/bulldozers-raw')       # replaced:  df_raw = pd.read_feather('tmp/bulldozers-raw')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "wv6n5K9xJYWW"
   },
   "source": [
    "We'll replace categories with their numeric codes, handle missing continuous values, and split the dependent variable into a separate variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "RMNGYeeTJYWt"
   },
   "outputs": [],
   "source": [
    "df, y, nas = proc_df(df_raw, 'SalePrice')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "qBxeK07uJYW-"
   },
   "source": [
    "We now have something we can pass to a random forest!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "MFN_5tOuJYXF"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=10, n_jobs=-1)\n",
    "m.fit(df, y)\n",
    "m.score(df,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "UbF5tn2UJYXq"
   },
   "source": [
    "In statistics, the coefficient of determination, denoted R2 or r2 and pronounced \"R squared\", is the proportion of the variance in the dependent variable that is predictable from the independent variable(s). https://en.wikipedia.org/wiki/Coefficient_of_determination"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "AaA-m9dUJYX5"
   },
   "source": [
    "Wow, an r^2 of 0.98 - that's great, right? Well, perhaps not...\n",
    "\n",
    "Possibly **the most important idea** in machine learning is that of having separate training & validation data sets. As motivation, suppose you don't divide up your data, but instead use all of it.  And suppose you have lots of parameters:\n",
    "\n",
    "<img src=\"https://github.com/Giffy/fast.ai/blob/master/Machine%20Learning/images/overfitting2.png?raw=1\" alt=\"\" style=\"width: 70%\"/>\n",
    "<center>\n",
    "[Underfitting and Overfitting](https://datascience.stackexchange.com/questions/361/when-is-a-model-underfitted)\n",
    "</center>\n",
    "\n",
    "The error for the pictured data points is lowest for the model on the far right (the blue curve passes through the red points almost perfectly), yet it's not the best choice.  Why is that?  If you were to gather some new data points, they most likely would not be on that curve in the graph on the right, but would be closer to the curve in the middle graph.\n",
    "\n",
    "This illustrates how using all our data can lead to **overfitting**. A validation set helps diagnose this problem."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "PHG_algGJYX_"
   },
   "outputs": [],
   "source": [
    "def split_vals(a,n): return a[:n].copy(), a[n:].copy()\n",
    "\n",
    "n_valid = 12000  # same as Kaggle's test set size\n",
    "n_trn = len(df)-n_valid\n",
    "raw_train, raw_valid = split_vals(df_raw, n_trn)\n",
    "X_train, X_valid = split_vals(df, n_trn)\n",
    "y_train, y_valid = split_vals(y, n_trn)\n",
    "\n",
    "X_train.shape, y_train.shape, X_valid.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "KyNpLEnwJYYl"
   },
   "source": [
    "# 3 Random Forests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "1fCdN9sYJYYr"
   },
   "source": [
    "## 3.1 Base model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "VA2EjSaeJYYx"
   },
   "source": [
    "Let's try our model again, this time with separate training and validation sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "k4ad6aWJJYY2"
   },
   "outputs": [],
   "source": [
    "def rmse(x,y): return math.sqrt(((x-y)**2).mean())\n",
    "\n",
    "def print_score(m):\n",
    "    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),\n",
    "                m.score(X_train, y_train), m.score(X_valid, y_valid)]\n",
    "    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)\n",
    "    print(res)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "7lrTnDLxJYZO"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=10, n_jobs=-1)\n",
    "%time m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "lg7dFGZxJYZ-"
   },
   "source": [
    "An r^2 in the high-80's isn't bad at all (and the RMSLE puts us around rank 100 of 470 on the Kaggle leaderboard), but we can see from the validation set score that we're over-fitting badly. To understand this issue, let's simplify things down to a single small tree."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "MM3lRHzEJYaE"
   },
   "source": [
    "## 3.2 Speeding things up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "D64lumPlJYaI"
   },
   "outputs": [],
   "source": [
    "df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)\n",
    "X_train, _ = split_vals(df_trn, 20000)\n",
    "y_train, _ = split_vals(y_trn, 20000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "bAa2ysb4JYaY"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=10, n_jobs=-1)\n",
    "%time m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "INHLVZWpJYa0"
   },
   "source": [
    "## 3.3 Single tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "4BxPUnDjJYa9"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "sBfG4z4gJYbV"
   },
   "outputs": [],
   "source": [
    "draw_tree(m.estimators_[0], df_trn, precision=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "JW6qvVADJYbq"
   },
   "source": [
    "Let's see what happens if we create a bigger tree."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "IWsqPDz0JYbv"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "7JzioIUMJYcG"
   },
   "source": [
    "The training set result looks great! But the validation set is worse than our original model. This is why we need to use *bagging* of multiple trees to get more generalizable results."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "U7lwOn9hJYcM"
   },
   "source": [
    "## 3.4 Bagging"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "i-Sz6CkzJYcT"
   },
   "source": [
    "### 3.4.1 Intro to bagging"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "DrnV38jsJYcZ"
   },
   "source": [
    "To learn about bagging in random forests, let's start with our basic model again."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "gboErTyZJYcg"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=10, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "DjlhPpL0JYdH"
   },
   "source": [
    "We'll grab the predictions for each individual tree, and look at one example."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "v7EVLKBcJYdc"
   },
   "outputs": [],
   "source": [
    "preds = np.stack([t.predict(X_valid) for t in m.estimators_])\n",
    "preds[:,0], np.mean(preds[:,0]), y_valid[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "JRBjL4dSJYeC"
   },
   "outputs": [],
   "source": [
    "preds.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "a1yt4h7uJYeZ"
   },
   "outputs": [],
   "source": [
    "plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "FG1ekzuCJYe2"
   },
   "source": [
    "The shape of this curve suggests that adding more trees isn't going to help us much. Let's check. (Compare this to our original model on a sample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "tngMIOu8JYe8"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=20, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "KOfarYXpJYfm"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "b-uCXImbJYgI"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=80, n_jobs=-1)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "J1TeAdSjJYgl"
   },
   "source": [
    "### 3.4.2 Out-of-bag (OOB) score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "RgDre3LWJYgq"
   },
   "source": [
    "Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both? With the existing information we've shown, we can't tell. However, random forests have a very clever trick called *out-of-bag (OOB) error* which can handle this (and more!)\n",
    "\n",
    "The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was *not* included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.\n",
    "\n",
    "This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.\n",
    "\n",
    "This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our `print_score` function below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "KbXzNPQyJYg0"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "ZYnQINmsJYhH"
   },
   "source": [
    "This shows that our validation set time difference is making an impact, as is model over-fitting."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "GDFmuGyaJYhM"
   },
   "source": [
    "## 3.5 Reducing over-fitting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "OHaKWiOlJYhQ"
   },
   "source": [
    "### 3.5.1 Subsampling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "kchoDwJ3JYhU"
   },
   "source": [
    "It turns out that one of the easiest ways to avoid over-fitting is also one of the best ways to speed up analysis: *subsampling*. Let's return to using our full dataset, so that we can demonstrate the impact of this technique."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "VrzLVZNzJYhZ"
   },
   "outputs": [],
   "source": [
    "df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')\n",
    "X_train, X_valid = split_vals(df_trn, n_trn)\n",
    "y_train, y_valid = split_vals(y_trn, n_trn)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "vsxjEuwwJYiE"
   },
   "source": [
    "The basic idea is this: rather than limit the total amount of data that our model can access, let's instead limit it to a *different* random subset per tree. That way, given enough trees, the model can still see *all* the data, but for each individual tree it'll be just as fast as if we had cut down our dataset as before."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "exdTdQiaJYiP"
   },
   "outputs": [],
   "source": [
    "set_rf_samples(20000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ZchmCZJ7JYig"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=10, n_jobs=-1, oob_score=True)\n",
    "%time m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "fPboiztCJYiz"
   },
   "source": [
    "Since each additional tree allows the model to see more data, this approach can make additional trees more useful."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "CU8EUwFdJYi7"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "eG0Nmn9jJYjW"
   },
   "source": [
    "### 3.5.2 Tree building parameters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "WSgmsndHJYjj"
   },
   "source": [
    "We revert to using a full bootstrap sample in order to show the impact of other over-fitting avoidance methods."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "8USFlC0NJYjq"
   },
   "outputs": [],
   "source": [
    "reset_rf_samples()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "_c3SMaBlJYj4"
   },
   "source": [
    "Let's get a baseline for this full set to compare to."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "GwA78_1zJYj8"
   },
   "outputs": [],
   "source": [
    "def dectree_max_depth(tree):\n",
    "    children_left = tree.children_left\n",
    "    children_right = tree.children_right\n",
    "\n",
    "    def walk(node_id):\n",
    "        if (children_left[node_id] != children_right[node_id]):\n",
    "            left_max = 1 + walk(children_left[node_id])\n",
    "            right_max = 1 + walk(children_right[node_id])\n",
    "            return max(left_max, right_max)\n",
    "        else: # leaf\n",
    "            return 1\n",
    "\n",
    "    root_node_id = 0\n",
    "    return walk(root_node_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "fPUiA31JJYkw"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "7FrKCP_NJYlT"
   },
   "outputs": [],
   "source": [
    "t=m.estimators_[0].tree_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "JDS7NM5YJYlm"
   },
   "outputs": [],
   "source": [
    "dectree_max_depth(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "X1YMltBCJYl9"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "CTmQ__NMJYmZ"
   },
   "outputs": [],
   "source": [
    "t=m.estimators_[0].tree_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "KxCSbAxTJYnA"
   },
   "outputs": [],
   "source": [
    "dectree_max_depth(t)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "KxjK4DW-JYnz"
   },
   "source": [
    "Another way to reduce over-fitting is to grow our trees less deeply. We do this by specifying (with `min_samples_leaf`) that we require some minimum number of rows in every leaf node. This has two benefits:\n",
    "\n",
    "- There are less decision rules for each leaf node; simpler models should generalize better\n",
    "- The predictions are made by averaging more rows in the leaf node, resulting in less volatility"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Ch_LF1qTJYoP"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Hs2PSh67JYpV"
   },
   "source": [
    "We can also increase the amount of variation amongst the trees by not only use a sample of rows for each tree, but to also using a sample of *columns* for each *split*. We do this by specifying `max_features`, which is the proportion of features to randomly select from at each split."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "LRCAe19TJYps"
   },
   "source": [
    "- None\n",
    "- 0.5\n",
    "- 'sqrt'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "2dODgPj3JYqD"
   },
   "source": [
    "- 1, 3, 5, 10, 25, 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "bFx5tN8xJYqv"
   },
   "outputs": [],
   "source": [
    "m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)\n",
    "m.fit(X_train, y_train)\n",
    "print_score(m)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "-hNtnxjuJYrJ"
   },
   "source": [
    "We can't compare our results directly with the Kaggle competition, since it used a different validation set (and we can no longer to submit to this competition) - but we can at least see that we're getting similar results to the winners based on the dataset we have.\n",
    "\n",
    "The sklearn docs [show an example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) of different `max_features` methods with increasing numbers of trees - as you see, using a subset of features on each split requires using more trees, but results in better models:\n",
    "![sklearn max_features chart](https://raw.githubusercontent.com/Giffy/fast.ai/master/Machine%20Learning/imgs/sphx_glr_plot_ensemble_oob_001.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "So6S-giXlCio"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "NpFfOmXtJX-8",
    "Vo-cGVcxJX_u",
    "J-A6Fj4PJYAk",
    "tzXIKZlNJYBH",
    "rCsFj7OsJYBm",
    "afyeBraRJYGz",
    "0HC1kO4HJYIi",
    "MXWHp01OJYIx",
    "SUGOtVXkJYNq",
    "DhcxnrqRJYV6",
    "1fCdN9sYJYYr",
    "MM3lRHzEJYaE",
    "INHLVZWpJYa0",
    "i-Sz6CkzJYcT",
    "J1TeAdSjJYgl",
    "OHaKWiOlJYhQ",
    "eG0Nmn9jJYjW"
   ],
   "include_colab_link": true,
   "name": "lesson1-randomforest.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
