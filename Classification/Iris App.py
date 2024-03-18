#------------------ pip installs
#pip install streamlit
#pip install protobuf // #pip install --upgrade protobuf

#------------------ run streamlit
# streamlit run "<python file>"
# streamlit run "C:\Users\c17527k\Documents\My Experiments\Python Scripts\Iris App.py"

#------------------ load packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#------------------ Elements on the web app
#----- DF
st.title('Welcome to Streamlit!')
st.subheader('Mini dataframe')

st.write(
pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
    })
)

#----- Select box
st.subheader('Select box')
option = st.selectbox('Are you happy?',
                    ('Yes', 'No')
                    )

st.write('You selected:', option)


#----- Check box
st.subheader('Check box')
st.write('Are you happy?')
checkbox_one = st.checkbox('Yes')
checkbox_two = st.checkbox('No')

if checkbox_one and checkbox_two:
    value = 'Both.'
elif checkbox_one:
    value = 'Yes'
elif checkbox_two:
    value = 'No'
else:
    value = 'Nothing'

st.write(f'You selected: {value}')

#----- Line chart
st.subheader('Line chart')
# 10 * 2 dimensional data
chart_data = pd.DataFrame(
    np.random.randn(10, 2),
    columns=[f"Col{i+1}" for i in range(2)]
)

st.line_chart(chart_data)

#----- Plotly pie chart
st.subheader('Pie chart')
fig = go.Figure(
    data=[go.Pie(
        labels=['A', 'B', 'C'],
        values=[30, 20, 50]
    )]
)
fig = fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value',
    textfont_size=15
)

st.plotly_chart(fig)



#----- ML: iris prediction
st.subheader('Iris prediction')
iris_data = load_iris() # dim 150 x 4 # https://archive.ics.uci.edu/ml/datasets/iris
# separate the data into features and target
features = pd.DataFrame(
    iris_data.data, columns=iris_data.feature_names
)
target = pd.Series(iris_data.target)

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target
)


class StreamlitApp: # instance of the class StreamlitApp

    def __init__(self):
        self.model = RandomForestClassifier()

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Iris Data Classification</p>',
            unsafe_allow_html=True
        )
        sepal_length = st.sidebar.selectbox(
            f"Select {cols[0]}",
            sorted(features[cols[0]].unique())
        )

        sepal_width = st.sidebar.selectbox(
            f"Select {cols[1]}",
            sorted(features[cols[1]].unique())
        )

        petal_length = st.sidebar.selectbox(
            f"Select {cols[2]}",
            sorted(features[cols[2]].unique())
        )

        petal_width = st.sidebar.selectbox(
            f"Select {cols[3]}",
            sorted(features[cols[3]].unique())
        )
        values = [sepal_length, sepal_width, petal_length, petal_width]

        return values

    def plot_pie_chart(self, probabilities):
        fig = go.Figure(
            data=[go.Pie(
                    labels=list(iris_data.target_names),
                    values=probabilities[0]
            )]
        )
        fig = fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value',
            textfont_size=15
        )
        return fig

    def construct_app(self): # contruct_app method of the class is invoked

        self.train_data() # trains the iris data with random forest classifier
        values = self.construct_sidebar() # constructs the sidebar/panel 

        values_to_predict = np.array(values).reshape(1, -1) # fetches the customization selections made in the sidebar

        prediction = self.model.predict(values_to_predict)
        prediction_str = iris_data.target_names[prediction[0]]
        probabilities = self.model.predict_proba(values_to_predict)

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Iris Data Predictions </p>',
            unsafe_allow_html=True
        )

        #column_1, column_2 = st.beta_columns(2)
        column_1, column_2 = st.columns(2) # two cols to display the prediction and probability side-by-side
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")

        fig = self.plot_pie_chart(probabilities) # plot the probabilities
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

        return self


sa = StreamlitApp()
sa.construct_app()

