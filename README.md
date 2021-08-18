# EUBOPA
European Basket Option Pricing Approaches

## Scope
This project is designed to fulfil the assessment requirements for the Machine Learning Engineer post at Multiverse 
Computing.

## Objective
The objective of this assessment task is to develop three different approaches from three mathematical domains that can 
be used to value a stock option and then compare the results.

## Problem Statement
The client wants to explore novel methodologies for accelerating the pricing of exotic derivatives.
A neural network is to be designed for pricing the European Basket options under Black-Scholes assumptions. Once done, 
the prices predicted by the trained model are to be compared with a traditional numerical solver for option pricing. 
Lastly, the results of both the strategies are to be compared against an analytical solution for a single asset.

## Definitions of Financial References
* ### Black-Scholes Formula

    This is a globally acceptable model of Quantitative Finance to price the derivatives under some fixed assumptions. 
    This formula provides a closed-form solution to the price of an option hence making it analytical in nature.

* ### Black_Scholes Assumptions

  1. Markets are fair. No transcation costs and no hiccups.
  2. The under-lying asset follows the log-normal distribution.
  3. Volatility and Risk-Free interest rate are constant and known.
  4. This is valid only for European options which can only be exercised on the expiration date.
  5. No dividends are paid out during the life of the option.
                                                           
    
    The above assumptions are also valid for the implementation of this project

* ### geomteric Brownian Motion (gBM)
    A mathematical model that is used to predict the future paths of an under-lying asset given that asset is following
    a Brownian Motion. In simple terms, this model helps predict the random walk of a stock price.

* ### Monte Carlo Method
    This is a numerical method that is used to foresee the life of an asset or a stock that is following a gBM.
* ### Europan Basket Option
    Basket options are based on more than one underlying assets which can only be exercised on the date of expiration. 
    The payoff of a basket option is essentially the weighted average of all underlying assets.  

## 1. Introduction

Option price prediction is the “act of determining” the buy or sell value of a stock option traded on an exchange. The 
successful prediction of a option price could yield significant profit.
In this project, I have proposed a option pricing model using Artificial Neural Networks. This technique utilises 
**five** distinct features as the input parameters for training, and gives ‘Call Price’ for a European stock as the 
output.

## 2.  Data set

The requirement was to design a model for the pricing of an **European Basket Option**. After a lot of research, I 
found a raw data of stock options on [Kaggle](https://www.kaggle.com/bendgame/options-market-trades). 

### 2.1. Data set Description

There was no metadata attached to this dataset. I found following information while exploring this data.

         Entity                              |                    Description
         ------------------------------------|-----------------------------------------------------------------
         Total observations in the data      |                      62795
         Total trade symbols in the data     |                      2346
         Highest observation in all the data | [SPY (SPDR S&P 500 Trust ETF)](https://finance.yahoo.com/quote/SPY/)
         Total Count of SPY observations     |                      4455

As being one of the most frequent stock in the given data, I chose to work with SPY symbol. Furthermore, I selected only
the call options for the SPY dataset.

## 3. Assumptions attached to this project

1. The first assumption undertaken is that the data of SPY company is representing a weighted average of multiple 
underlying assets making it a basket option.
2. The option will only be exercised on the expiration date hence it is a European option.
3. **Volatility**, ```v```, is calculated by the following formula.

![volatility](assets/volatility.png)

where;

* sigma is the standard deviation 
* ```N``` = total no. of unique days in which the trading of asset of interest happened 

Using this formula, the volatility was fixed at ```0.031802533217352```

4. Adherence to the assumption no. 3 makes the whole project to be operating in the Black-Scholes world. 
5. Risk-Free Interest Rate is set as the 10-year US treasury yield from Yahoo Finance API. At the time of data processing 
   for this project, the risk-free interest rate was ```0.012580000162125```


## 4. ANN Model
ANNs are composed of multiple nodes, which imitate biological neurons of human brain. The neurons are connected by links
and they interact with each other. The nodes can take input data and perform simple operations on the data. The result 
of these operations is passed to other neurons. The output at each node is called its activation or node value.Each 
link is associated with weight.

For this project, a simple ANN architecture was designed using Keras API with TensorFlow backend. Model summary is given
as follows.

![model summary](assets/model_summary.png)
                                                                

## ANN Architecture

![architecture](assets/nn.svg)  

