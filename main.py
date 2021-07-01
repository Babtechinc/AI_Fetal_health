import streamlit as st

import time
import base64
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# model selection libraries
from sklearn.model_selection import train_test_split

# machine learning libraries
from sklearn.ensemble import RandomForestClassifier

# postprocessing and checking-results libraries
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

st.set_option('deprecation.showPyplotGlobalUse', False)
LOGO_IMAGE = 'check copy.png'

def load():
    """Simple function to imitate a progress bar"""
    columns = [str(i) for i in range(100)]
    columns = st.beta_columns(100)
    c = 0
    for val in columns[2:97]:
        with val:
            if c > 93:
                st.write(".")
            else:
                st.write("_")
        time.sleep(0.05)
        c += 1

#heading html
components.html(
    """
    <style>
    /* Style the body */
body {
  font-family: sans-serif;
  margin: 0;
}
hr {
  color: black;
  border: solid 1px black;
}

/* Header/Logo Title */
.header {
  padding: 15px;
  height:50px;
  width: 100%;
  margin: auto;
  text-align: center;
  background: #710E29;
  border: solid 1px transparent;
  border-radius: 10px;
}
.header_text {
  color: white;
  font-size: 20px;
}
.sub_header {
  color: black;
  font-size: 15px;
  text-align: center;
}

/* Page Content */
.content {padding:20px;}
    </style>

    <div class="header">
      <h1 class="header_text">
        FETAL HEALTH CLASSIFICATION EXPERT STSTEM - (BEGME)
      </h1>
    </div>
    <div>
      <h5 class="sub_header"> 
        Classify the health of a fetus as Normal, Suspect or Pathological using CTG data
      </h5>
    </div>
    <hr>
    <b>1. File Upload </b>
    <hr>
    """,
    height=200,
)

#Adding the upload button
left_column, right_column= st.beta_columns(2)
with left_column:
    uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataupload = pd.read_csv(uploaded_file)
    with left_column:
        components.html(
            """
            <hr>
            <i style="color:red"> Currently viewing Uploaded data: </i>
            <hr>
            """,
            height=50,
        )

    #UPLOAD SUCCESSFUL INDICATOR HTML
    with right_column:
        st.markdown(
            """
                <style>
                  body{
                  color: green;
                  height: 200px;
                  }
                  .text{color: green}
                  .container{
                  text-align: center;
                  margin-left: 50px;
                  border: solid 2px green;
                  border-radius: 10px}
                </style>
                    """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <br>
            <div class="container">
                <img src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
                <label class="text">Upload Successful</label>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1)

    st.write(dataupload)

    components.html(
        """
        <hr>
        """,
        height=15,
    )

    # Rendering the "CLASSIFY" button
    left_column, middle, r= st.beta_columns(3)
    with middle:
        button = st.button("CLASSIFY RESPECTIVE FETAL HEALTH")

    # Adding functionality to the "CLASSIFY" button
    if button:
        load()
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.size'] = 12


        def plotConfusionMatrix(dtrue, dpred, classes, title='Confusion Matrix', width=0.75, cmap=plt.cm.Blues):
            cm = confusion_matrix(dtrue, dpred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(np.shape(classes)[0] * width, np.shape(classes)[0] * width))
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes,
                   yticklabels=classes,
                   title=title,
                   aspect='equal')

            ax.set_ylabel('True', labelpad=20)
            ax.set_xlabel('Predicted', labelpad=20)

            plt.setp(ax.get_xticklabels(), rotation=90, ha='right',
                     va='center', rotation_mode='anchor')

            fmt = '.2f'

            thresh = cm.max() / 2.0

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                            color='white' if cm[i, j] > thresh else 'black')
            plt.tight_layout()
            plt.show()

        # Loading dataset for training
        df = pd.read_csv('fetal_health.csv')
        df.drop_duplicates(inplace=True)

        #Feature selection (adjust target labels to start from 0 and drop target label from input values(X))
        y = LabelEncoder().fit_transform(df['fetal_health'])
        X = df.drop(columns=['fetal_health'], axis=1)

        #Initial Exploratory Data Analysis(EDA)
        count = np.zeros(3)
        for i in range(3):
            count[i] = np.where(y == i)[0].size

        plt.subplots(figsize=(6.0, 6.0))
        plt.bar(np.arange(3), count, color='orange', edgecolor='black')
        plt.xticks(np.arange(3), ('N', 'S', 'P'))
        plt.xlabel('Fetal State')
        plt.ylabel('Number of Instances')

        #Scaling the data (Not really needed for Random Forests Algorithm)
        # scaler = StandardScaler().fit(X)
        # Xnorm = scaler.transform(X)

        #Splitting dataset into training and testing data
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True,
                                                        random_state=21)

        #OverSampling to deal with data imbalance
        Xtrain, ytrain = RandomOverSampler(random_state=21).fit_resample(Xtrain, ytrain)

        #Training the model
        clf = RandomForestClassifier(random_state=21).fit(Xtrain, ytrain)

        #Applying model on test data
        ypred = clf.predict(Xtest)

        # Testing the model with uploaded data (by user)
        dataupload.drop_duplicates(inplace=True)
        NewX = dataupload
        Newypred = clf.predict(NewX) #feeding trained model with uploaded data
        Newypred = Newypred + 1 #Reversing effect of LabelEncoder
        Newpreddf = pd.DataFrame(Newypred) #convert Newpred to a dataframe from a numpy array
        NewX.insert(0, 'predicted_fetal_health', Newpreddf) #add new column (predicted_fetal_health) to dataset
        NewX['predicted_fetal_health'] = NewX['predicted_fetal_health'].replace([1, 2, 3], ['Normal', 'Suspect', 'Pathological'])

        #Results html
        components.html(
            """
            <style>
            /* Style the body */
        body {
          font-family: sans-serif;
          margin: 0;
        }
        hr {
          color: black;
          border: solid 1px black;
        }
        /* Page Content */
        .content {padding:20px;}
            </style>
            <hr>
            <b>2. Results </b>
            <hr>
            """,
            height=60,
        )

        # Print prediction output
        st.write(NewX)
        # st.write(Newypred) these are the predicted fetal healths
        # st.write(y[:49]) outputs first 49 rows of the actual fetal healths(from training data)

        #Accuracy/Performance html
        components.html(
            """
            <style>
            /* Style the body */
        body {
          font-family: sans-serif;
          margin: 0;
        }
        hr {
          color: black;
          border: solid 1px black;
        }
        /* Page Content */
        .content {padding:20px;}
            </style>
            <hr>
            <b>3. Accuracy/Performance metrics </b>
            <hr>
            """,
            height=60,
        )

        #Display accuracy metrics and graphs
        left_column, right_column = st.beta_columns(2)
        with left_column:
            st.write(classification_report(y[:50], Newypred - 1))
            # st.write(classification_report(ytest, ypred))
        with right_column:
            st.pyplot(plotConfusionMatrix(y[:50], Newypred - 1, classes=np.array(['N', 'S', 'P']), width=1.5, cmap=plt.cm.binary))
