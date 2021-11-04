import streamlit as st
import pandas as pd
import hdp



def main():
	st.write("""
		# HEART DISEASE PREDICTION
		Hello, *welcome from heart disease predicting app!*
		""" 
		"""
		Heart disease is the leading cause of death in the United States, causing about 1 in 4 deaths. 
		The term “heart disease” refers to several types of heart conditions. In the United States, 
		the most common type of heart disease is coronary artery disease (CAD), which can lead to heart attack.
		"""
		"""
		#### This app is to predict whether a person has heart disease or not.
		*Please fill up some data to see result.*
		Here is an example of some data.
		""")

	df=pd.read_csv('heart.csv')
	df

	st.write("""###### Enter your age""")
	number_input=st.number_input("Enter your age",min_value=0,max_value=200)

	st.write("""###### Choose sex""")
	number_input=st.number_input("Pick your gender(if male, type 1, if female, type 0.)",min_value=0,max_value=1)

	st.write("""###### What is your cp?""")
	number_input=st.number_input("Enter your cp",min_value=0,max_value=10)

	st.write("""###### Trestbps""")
	number_input=st.number_input("Enter your Trestbps",min_value=100,max_value=200)

	st.write("""###### Chol""")
	number_input=st.number_input("Enter your chol",min_value=0,max_value=400)

	st.write("""###### Fbs""")
	number_input=st.number_input("fbs?(if it's under type 0 or else type 1)",min_value=0,max_value=1)
	st.write("""###### Restecg""")
	number_input=st.number_input("restecg?(if true type 1, if false type 0",min_value=0,max_value=1)

	st.write("""###### Thalach""")
	number_input=st.number_input("Enter your thalach",min_value=100,max_value=200)

	st.write("""###### Exang""")
	number_input=st.number_input("exang?(if true type 1, if false type 0)",min_value=0,max_value=1)

	st.write("""###### Oldpeak""")
	number_input=st.number_input("oldpeak?")

	st.write("""###### slope""")
	number_input=st.number_input("slope:",min_value=0,max_value=2)

	st.write("""###### CA""")
	number_input=st.number_input("CA?",min_value=0,max_value=10)

	st.write("""###### THAL""")
	number_input=st.number_input("Enter your THAL",min_value=0,max_value=3)


	if st.button("Predict"):
		result=hdp.predicting(number_input)
	st.success(hdp.predicting(number_input))


if __name__=='__main__':
	main()


