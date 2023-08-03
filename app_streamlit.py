import pandas as pd
import numpy as np
import streamlit as st
import nltk
import torch
from transformers import AutoTokenizer, DefaultDataCollator, BertForPreTraining

from src.utilities import clean_output_text

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
data_collator = DefaultDataCollator(return_tensors="pt")
bert_model = BertForPreTraining.from_pretrained("nlpaueb/sec-bert-base")
model = torch.load(f"data/models/feature_based_sen/NN/2023-03-26_0946_MDuA_all/epoch_2.pt",
                   map_location=torch.device('cpu'))

examples = {
    'pos_example_1': {
        'company': 'Reddy Ice',
        'fiscal_year': 2011,
        'url': 'https://www.sec.gov/Archives/edgar/data/1268984/000104746912004191/a2208578z10-k.htm',
        'headline': 'Restructuring Transactions - Chapter 11 Bankruptcy Cases',
        'paragraph':
            """
            In order to consummate the transactions described below, including the entry into the DIP Credit Facility, the amendment to the First Lien Notes, the exchange of the Second Lien Notes, the cancellation of the Discount Notes, and the consummation of the rights offering, we intend to commence the solicitation of acceptances with respect to a plan of reorganization (the "Plan") pursuant to this Disclosure Statement. Shortly after commencing such solicitation, Reddy Ice Holdings, Inc. and Reddy Ice Corporation intend to file voluntary bankruptcy cases under Chapter 11 of the U.S. Bankruptcy Code (the "Bankruptcy Cases") in the United States Bankruptcy Court for the Northern District of Texas (the "Bankruptcy Court"). This process could, but is not expected to, have an adverse effect on our operations. Immediately upon filing, we will request Bankruptcy Court approval to pay critical suppliers and other vendors in the ordinary course of business. All of our customers will continue to be serviced without interruption. Although we intend to continue to provide such products and services to our customers in the normal course, there can be no assurances that counterparties will continue to conduct business with us while we are in Chapter 11.
            With respect to the Plan, we will be soliciting acceptances from only the holders of the First Lien Notes, the Second Lien Notes (and their respective guarantees), the Discount Notes and certain other unsecured creditors of Reddy Holdings, although the Plan will seek Bankruptcy Court approval of recoveries to holders of other unsecured creditors and the Common Stock as well. The deadline for voting to accept or reject the Plan is May 8, 2012, unless such deadline is extended. In addition, we will request the Bankruptcy Court to set a confirmation hearing with respect to the Plan for early May 2012 and hope to exit protection under the Bankruptcy Code in the second or third week of May—before final round bids must be submitted with respect to the proposed Arctic acquisition.
            Support agreements for the Plan have been executed by holders of approximately 60% of the principal amount of the First Lien Notes, approximately 58% of the principal amount of the Second Lien Notes and approximately 92% of the principal amount of the Discount Notes. Such support agreements require the holders of the First Lien Notes, the Second Lien Notes and the Discount Notes to, among other things, vote to accept the Plan.
            """,
    },
    'pos_example_2': {
        'company': 'Unit Corporation',
        'fiscal_year': 2019,
        'url': 'https://www.sec.gov/Archives/edgar/data/798949/000079894920000010/unt-20191231.htm',
        'headline': 'Executive Summary - Oil and Natural Gas',
        'paragraph':
            """
            Fourth quarter 2019 production from our oil and natural gas segment was 4,157 MBoe, a decrease of 5% and 4% from the third quarter of 2019 and the fourth quarter of 2018, respectively. The decreases came from fewer net wells being completed in the fourth quarter to replace declines in previously drilled wells. Oil and NGLs production during the fourth quarter of 2019 was 48% of our total production compared to 46% of our total production during the fourth quarter of 2018.
            Fourth quarter 2019 oil and natural gas revenues increased 7% over the third quarter of 2019 and decreased 21% from the fourth quarter of 2018. The increase over the third quarter of 2019 was primarily due an increase in commodity prices partially offset by a decrease in equivalent production. The decrease from the fourth quarter of 2018 was primarily due to a decrease in commodity prices and equivalent production.
            Our hedged natural gas prices for the fourth quarter of 2019 increased 8% over third quarter of 2019 and decreased 29% from fourth quarter of 2018. Our hedged oil prices for the fourth quarter of 2019 increased 1% and 6% over the third quarter of 2019 and the fourth quarter of 2018, respectively. Our hedged NGLs prices for the fourth quarter of 2019 increased 54% over the third quarter of 2019 and decreased 33% from fourth quarter of 2018.
            Direct profit (oil and natural gas revenues less oil and natural gas operating expense) increased 24% over the third quarter of 2019 and decreased 29% from the fourth quarter of 2018. The increase over the third quarter of 2019 was primarily due to an increase in commodity prices and a reduction in lease operating expenses (LOE) and general and administrative (G&A) expenses partially offset by a decrease in equivalent production. The decrease from the fourth quarter of 2018 was primarily due to lower revenues due to lower commodity prices and volumes and higher salt water disposal expenses and gross production taxes.
            Operating cost per Boe produced for the fourth quarter of 2019 decreased 8% from the third quarter of 2019 and increased 3% over the fourth quarter of 2018. The decrease from the third quarter of 2019 was primarily due to lower G&A and LOE. The increase over the fourth quarter of 2018 was primarily due to increased saltwater and production taxes along with lower equivalent production.
            In our Wilcox play, located primarily in Polk, Tyler, Hardin, and Goliad Counties, Texas, we completed seven vertical gas/condensate wells (average working interest 100%) in 2019. Annual production from our Wilcox play averaged 76 MMcfe per day (7% oil, 21% NGLs, 72% natural gas) which is a decrease of 15% compared to 2018. We averaged approximately 0.75 Unit drilling rigs operating during 2019.
            In our Southern Oklahoma Hoxbar Oil Trend (SOHOT) play in western Oklahoma, primarily in Grady County, we completed seven horizontal oil wells in the Marchand zone of the Hoxbar interval and, in our Red Fork play, we completed seven horizontal wells. Average working interest for these wells was 85.3%. Annual production from western Oklahoma averaged 95.7 MMcfe per day (35% oil, 22% NGLs, 43% natural gas) which is an increase of approximately 25% compared to 2018. During 2019, we averaged approximately 1.5 Unit drilling rigs operating and we participated in 61 non-operated wells in the mid-continent region, with most of those occurring in the STACK play. Unit’s average working interest in these non-operated wells is 3.7%.
            In our Texas Panhandle Granite Wash play, we completed two extended lateral horizontal gas/condensate wells (average working interest 98.5%) in our Buffalo Wallow field. Annual production from the Texas Panhandle averaged 91.9 MMcfe per day (9% oil, 37% NGLs, 55% natural gas) which is a decrease of approximately 5% compared to 2018. We used 0.25 Unit drilling rigs during 2019.
            In December of 2019, we sold our Panola Field in eastern Oklahoma for $17.9 million.
            During 2019, we participated in the drilling of 115 wells (29.15 net wells). For 2020, we do not currently have any plans to drill wells pending our ability to refinance our debt.
            """,
    },
    'pos_example_3': {
        'company': 'Global Eagle Entertainment Inc.',
        'fiscal_year': 2019,
        'url': 'https://www.sec.gov/Archives/edgar/data/1512077/000151207720000006/ent-10kdocument4q19.htm',
        'headline': 'Liquidity and Capital Resources - Covenant Compliance Under 2017 Credit Agreement',
        'paragraph':
            """
            As of December 31, 2019, we were in compliance with all financial and non-financial covenants under the 2017 Credit Agreement, including the financial reporting and leverage ratio covenants. On April 15, 2020, the Company entered into the Tenth Amendment to the Credit Agreement and obtained a waiver related to obtaining a “going concern” or like qualification or exception opinion for the Company’s the year-end December 31, 2019 financial statements. Given the uncertainty of the COVID-19 impact on the Company’s results of operations and liquidity subsequent to December 31, 2019, we do not expect to remain in compliance with all financial covenants in the second half of 2020. We cannot be assured that we will be able to obtain additional covenant waivers or amendments in the future which may have a material adverse effect on the Company’s results of operations or liquidity.
            You should also refer to the section titled “Risks Related to Our Liquidity and Indebtedness” in Part I, Item 1A. Risk Factors in this Form 10-K, for an explanation of the consequences of our failure to satisfy these covenants. If we fail to satisfy the leverage ratio covenant, then an event of default under the 2017 Credit Agreement would occur. If the lenders thereunder fail to waive such default, then the lenders could elect (upon a determination by a majority of the lenders) to terminate their commitments and declare all amounts borrowed under 2017 Credit Agreement due and payable. This acceleration would also result in an event of default under the indenture governing our convertible notes and Second Lien Notes.
            Consolidated EBITDA as defined in the 2017 Credit Agreement is a non-GAAP financial measures that we use to determine our compliance with the maximum first lien leverage ratio covenant in the 2017 Credit Agreement. Consolidated EBITDA, calculated pursuant to the 2017 Credit Agreement, means net income (loss), calculated in accordance with GAAP, plus (a) total interest expense, (b) provision for taxes based on income, profits or capital gains, (c) depreciation and amortization and (d) other applicable items as set forth in the 2017 Credit Agreement.
            If we are unable to achieve the results required to comply with this covenant in one or more quarters over the next twelve months, we may be required to take specific actions in addition to those described above, including but not limited to, additional reductions in headcount and targeted procurement initiatives to reduce operating costs and, or alternatively, seeking a waiver or an amendment from our lenders. If we are unable to satisfy our financial covenants or obtain a waiver or an amendment from our lenders, or take other remedial measures, we will be in default under our credit facilities, which would enable lenders thereunder to accelerate the repayment of amounts outstanding and exercise remedies with respect to the collateral. If our lenders under our credit facilities demand payment, we will not have sufficient cash to repay such indebtedness. In addition, a default under our credit facilities or the lenders exercising their remedies thereunder could trigger cross-default provisions in our other indebtedness and certain other operating agreements. Our ability to amend our credit facilities or otherwise obtain waivers from our lenders depends on matters that are outside of our control and there can be no assurance that we will be successful in that regard. In addition, any covenant breach or event of default could harm our credit rating and our ability to obtain financing on acceptable terms. The occurrence of any of these events could have a material adverse effect on our financial condition and liquidity.
            """,
    },
    'pos_example_4': {
        'company': 'Edge Petroleum Corporation',
        'fiscal_year': 2008,
        'url': 'https://www.sec.gov/Archives/edgar/data/1021010/000104746909002774/a2191585z10-k.htm',
        'headline': 'Revenue and Production - Derivatives',
        'paragraph':
            """
            The volume and price contract terms of our derivative contracts vary from period to period and therefore interact differently with the changing pricing environment, which makes the comparability of the results for each period difficult. In all periods presented, we applied mark-to-market accounting treatment to our derivative contracts; therefore the full volatility of the non-cash change in fair value of our outstanding contracts is reflected in total revenue and will continue to affect total revenue until outstanding contracts expire. Since these gains/losses are not a function of the operating performance of our oil and natural gas assets, excluding their impact from the above discussions helps isolate the operating performance of those assets. The following table
            Should crude oil or natural gas prices increase or decrease from the current levels, it could materially impact our revenues. In a high price environment, hedged positions could result in lost opportunities if there is a cap in place, thus lowering our effective realized prices on hedged production, but in an environment of falling prices, these transactions offer some pricing protection for hedged production. Our overall 2008 derivative position exceeded our total 2008 production, due to changes in forecasted production and certain asset divestitures. Although we took steps in the fourth quarter of 2008 to reduce the overhedge exposure, the position exposed us to greater losses during those periods of high prices in the second and third quarters of 2008. For additional discussion of the overhedge position, see Note 9 to our consolidated financial statements.
            """,
    },
}

st.markdown('''
    # Demo: Leveraging MD&A Sentences and Unlocking Bankruptcy Clues

    This demo summarizes a text based on the likelihood that a company declares bankruptcy (chapter 7 or 11) within the next year. 
    The underlying approach is outlined in *Leveraging MD&A Sentences and Unlocking Bankruptcy Clues* (Hesse & Loy, 2023).

    The model for this demo is trained on all Management Discussion and Analysis (MD&A) documents in html format from the fiscal year 2001 to 2019. 
    The in-sample AUC score for this model is 0.69 on the sentence-level and 0.86 on the sentence-to-document-level. 
    The probability threshold for the summarization task is preset to the in-sample maximal F1 score.

    * Insert a sentence, paragraph or whole MD&A document into the text area or press insert on the preselected examples
    * Choose a probability cutoff (a higher value is more conservative and flags less sentences than a smaller value)
    * Press Process

    The preselected examples are the paragraphs which contain the correctly identified exemplary sentences indicating next year’s bankruptcy shown in Hesse & Loy (2023).
    The bankruptcy probability corresponding to a sentence may differ to what is written in Hesse & Loy (2023) since this model is only trained on sentence embeddings and does take advantage of financial and/or market information. 
''')

input_box = st.text_area('Enter text here...', height=200)

slider_value = st.slider('Threshold:', min_value=0.0, max_value=1.0, step=0.01, value=0.48)

button = st.button('Process')
reset_button = st.button('Reset')

if reset_button:
    input_box = ""

# Example buttons
st.markdown('## Examples')
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button('Insert Example 1'):
        input_box = clean_output_text(examples['pos_example_1']['paragraph'])
with col2:
    if st.button('Insert Example 2'):
        input_box = clean_output_text(examples['pos_example_2']['paragraph'])
with col3:
    if st.button('Insert Example 3'):
        input_box = clean_output_text(examples['pos_example_3']['paragraph'])
with col4:
    if st.button('Insert Example 4'):
        input_box = clean_output_text(examples['pos_example_4']['paragraph'])

if button:
    if not input_box:
        st.warning('No sentences found.')
    else:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Split text into sentences
        sentences = nltk.tokenize.sent_tokenize(input_box)

        # Tokenize sentences using Hugging Face's tokenizer
        encoded_sentences = tokenizer(sentences, padding='max_length', truncation=True, max_length=128)

        # Get embeddings
        bert_model.eval()
        with torch.no_grad():
            output = bert_model(torch.LongTensor(encoded_sentences['input_ids']),
                                attention_mask=torch.LongTensor(encoded_sentences['attention_mask']),
                                output_attentions=False, output_hidden_states=True)
            embeddings = output['hidden_states'][12][:, 0, :]

        # Raw predictions
        model.eval()
        pred = np.round(model(embeddings).detach().numpy(), 2)

        # Filter sentences based on slider value
        filtered_pred = [pred[i, 1] for i in range(len(sentences)) if pred[i, 1] >= slider_value]
        filtered_sentences = [sentences[i] for i in range(len(sentences)) if pred[i, 1] >= slider_value]
        df_output = pd.DataFrame({'Sentence': filtered_sentences, 'Prediction': filtered_pred})
        if len(filtered_sentences) == 0:
            st.warning('No sentences found above threshold.')
        else:
            st.table(df_output.style.format({"Prediction": "{:.2f}"}))