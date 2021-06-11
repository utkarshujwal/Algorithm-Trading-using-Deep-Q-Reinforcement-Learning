# Algorithm-Trading-using-Deep-Q-Reinforcement-Learning

## Introduction
Stock trading strategies play a critical role in investment. However, it is challenging to design a
profitable strategy in a complex and dynamic stock market. In this project, we propose a strategy
that employs deep reinforcement learning to learn a stock trading strategy by maximizing
investment return. We train a deep reinforcement learning agent and obtain a trading strategy
using Q values from deep Q-Network. This strategy trains the agent to take certain actions at
certain states within an environment to maximize rewards, thereby robustly adjusting to different
market situations. We test our algorithms on a pair of NASDAQ’s stocks that have adequate
liquidity. The performance of the trading agent is evaluated using the data of the same pair stocks
on different time periods. The proposed strategy is shown to outperform the benchmark model in
terms of the Net PnL for the whole period and Portfolio value.

![image](https://user-images.githubusercontent.com/47473472/119594285-1be04900-bda1-11eb-8eac-c7b88d09e2d6.png)


#### Problem Statement
Since building a trading agent that can choose from all the available stocks is a difficult problem,
we have chosen to start with a smaller number of stocks- 2 stocks.

We will use a reinforcement learning model to train an agent that will buy, sell or hold 2 stocks
everyday, once a day, so that the long term capital increases.

● Agent has an initial capital of $ X.

● Environment has 2 equities that an agent can trade A and B.

● Price of these equities are defined as P <sup>A</sup><sub>t</sub> and P<sup>B</sup><sub>t</sub>.

● The agent can perform tasks: Buy-A, Sell-A, Do Nothing, Buy-B and Sell-B.

To make the problem simpler, there are no transaction costs considered in buying/selling stocks.

#### Metrics
A basic metric used in stock trading is Profit and Loss. I will also use the same metric. Agent
starts with a capital of $100,000 and stocks of Apple and Amazon with $50,000 each. At the end
of episode, we calculate net PnL as follows:

● Net PnL (Net Profit and Loss) for the whole period = (Portfolio Value) at the end of
period - (Portfolio Value) at the start of the period.

● Portfolio Value= (Quantity of Apple Stock* Price of Apple Stock) + (Quantity of
Amazon Stock* Price of Amazon Stock) + Open Cash

#### Trading Strategy
The problem that we are trying to solve here is for a Day Trader and not a long-term Warren
Buffet like investment. Hence in this problem:

● We are not looking at fundamental analysis of stock price, but at a daily small variation
in stock price.

● We are also not taking the risk that is involved in taking a big position in a stock (so
buying/selling in lots of say 100, 1000, 500 stocks at the same time). We are trying to
find a daily buy/sell/hold strategy, with buying 1 stock a day out of the two available
stocks and then trying to make profit from this.

Therefore, this metric Profit and Loss is the best for this problem.

Past Work Done in this area is given in the reference section. Reinforcement learning will be
used to solve the above problem. I will be using a Deep Q-learning algorithm to solve the
problem. Detailed algorithm defined in the later sections.

## Exploratory Data Analysis

This data is gathered from kaggle:

https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/home

The data is presented in CSV format as follows: Date, Open, High, Low, Close, Volume,
OpenInt. We will use the following columns from the dataset: Date, Open, Close and Volume.
The data is for a period of more than 15 years (various stocks have various data points). We will
be planning to use 2 equities from the data set - Apple (aapl.us.txt) and Amazon (amzn.us.txt)

![image](https://user-images.githubusercontent.com/47473472/119594696-e425d100-bda1-11eb-931b-3fecd7763fa0.png) 
##### Figure 1: Price/Volume vs Year plot of Apple,Inc Equity

![image](https://user-images.githubusercontent.com/47473472/119594786-0cadcb00-bda2-11eb-9948-30b6c839a31b.png)
##### Figure 2: Price/Volume vs Year plot of Amazon,Inc Equity

Above are graphs of Apple and Amazon stocks from Year 1997 till 2017. From the graph, we
can see the following:

1. If an agent buys the stocks at the start of 1997 and just holds it for the whole period and sells it
at the end of the period, there will be a lot of profit. Our benchmark model does the same. So,
beating this benchmark model in itself will be a challenge for the agent.

2. The agent thus has to be trained to be in the game till the end of the period and sell when the
prices are high and buy when they get low, so that it beats the benchmark model.

3. There is not much correlation of prices and volume over the whole period. For e.g. during
Year 2000, volume was too high, but price was not that high.

## DQN Implementation
In our case, reinforcement learning agent can:

● At every state S <sub>t</sub> of environment (i.e. every day, with prices of Apple and Amazon stock
given).

● Take one action from possible actions At (Buy Apple, Buy Amazon, Buy Amazon, sell
Amazon, Do nothing).

● And Based on action taken, the agent gets a reward R<sub>t</sub>.

● Next State S <sub>(t+1)</sub> gets defined (i.e. new prices for next day).


![image](https://user-images.githubusercontent.com/47473472/119594965-60201900-bda2-11eb-948b-a3493b70b582.png)


## Benchmark Model

Benchmark model is a model where there is no agent, no reinforcement learning . So, the idea is
that we would like to compare my agent with a simple benchmark where there is no intelligencejust
buy the stocks at the start of the period and sell it periodically over the whole period, with
10% sell every time.

The benchmark model:

● At the start of the period: We buy stocks Apple and Stock Amazon, with half the amount
invested in each stock. Open Cash remains the same through-out the period.

● Sell 10% of stocks in 10 intervals, thus increasing open cash.

● At the end of period calculate portfolio value as follows:

   ○ Portfolio Value= (Quantity of Apple Stock* Price of Apple Stock) + (Quantity of Amazon Stock* Price of Amazon Stock) + Open Cash
   
## Design Choices for DQN Model

##### State/Environment

Not all variables are needed to train the agent, So we have to discard a few.

1. For Price, we have 3 variables: Open, High and Close Price. I chose Open price, as the
agent trades only once in a day, so start of the day. Other 2 prices were discarded from
the state.

2. We need the balance of Apple and Amazon stock every day.
3. We would also need Open Cash available for the agent to buy stocks.
4. Portfolio value at any point of time.
Based on various trial and error, we found that a 5-day trailing price of stocks will help the agent
in making the right choice about when it is a good action to buy/sell/do-nothing. This is a typical
parameter that even traders use to make a buy/sell decision.
State : Stock1Price, Stock2Price, Stock1Balance, Stock2Balance, OpenCash,
Stock1_5day_price, Stock2_5day_price and PortfolioValue

##### Agent

The agent is initialized with gamma, epsilon, memory size, action_size and state_size. Based on
the state of the environment, the agent either explores or exploits the state to gain the reward.
Actions: BuyStock1, BuyStock2, DoNothing, SellStock1or SellStock2.

## Main Functionality

![image](https://user-images.githubusercontent.com/47473472/119595715-94e0a000-bda3-11eb-81fd-d35ea2f53a4b.png)

● The DQN agent is trained with 100 episodes. For every 10 th episode, we are storing a
model that can be used during the testing phase. In each episode, we run the program for
all the days in the episode.

● We defined 2 variables, changepercentageStock1 and changepercentageStock2. These
variables are used to check whether today's stock price is higher or lower than the last
5-day stock price. Rewards for various actions are based on the value of these variables.

● These are the possible actions and associated rewards:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ○ Action 0: this action will buy the Apple stock if Cash is available

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Agent tries to buy Apple stock when it does not have enough Open
Cash, then the agent gets bankrupt, the episode ends and the reward is a
big negative number, -200000.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Open cash is less than 500 and agent tries to sell it, then give negative
reward -10000, this is to train agent to not go bankrupcy.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Also, if the stock price is only 2% less than the last 5-day trading price,
then the reward is -10,000. This is to ensure that agent does not do trading for small gains.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Otherwise, the agent gets a reward based on how much the buying price is
less than 5-day trailing price.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ○ Action 1 : in this action program will sell the Apple stock

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Agent has 0 Apple Stocks and it tries to buy it, then the agent gets
bankrupt, the episode ends and the reward is a big negative number -200000.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Apple stocks are less than 10 and an agent tries to sell it, then give a
negative reward -100000, this is to train the agent to not go bankrupt.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Also, if the stock price is only 2% less than the last 5-day trading price,
then the reward is -10,000. This is to ensure that the agent does not do trading for small gains.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Otherwise, the agent gets a reward based on how much the buying price is
less than 5-day trailing price.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ○ Action 2 : Agent will do nothing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If an agent has less than 2 Apple and Amazon stocks, and it does nothing,
then it is good and it gets a reward of 10000.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Else, it gets a reward of -100000 for not doing anything. This is to make
the agent take an action.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ○ Action 3 : this action will buy the Amazon stock if Cash is available.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Agent tries to buy Amazon stock when it does not have enough Open Cash, then the agent gets bankrupt, the episode ends and the reward is a big negative number -200000.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Open cash is less than 500o and the agent tries to sell it, then give a
negative reward -10000, this is to train the agent to not go bankrupt.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Also, if the stock price is only 2% less than the last 5-day trading price,
then the reward is -10,000. This is to ensure that the agent does not do trading for small gains.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Otherwise, the agent gets a reward based on how much the buying price is
less than 5-day trailing price.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ○ Action 4 : this action will sell the Amazon stock if Cash is available

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Agent has 0 Amazon Stocks and it tries to buy it, then the agent gets
bankrupt, the episode ends and the reward is a big negative number -200000.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ If Apple stocks are less than 10 and agent tries to sell it, then give negative
reward -10000, this is to train agent to not go bankrupt.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Also, if the stock price is only 2% less than the last 5-day trading price,
then the reward is -10,000. This is to ensure that the agent does not do
trading for small gains.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ■ Otherwise, the agent gets a reward based on how much the buying price is
less than 5-day trailing price.

## Experience Replay Memory
Our problem is that we have sequential samples from interactions with the environment to our
neural network. And it tends to forget the previous experiences as it overwrites with new
experiences.

For instance, if we are on Day 400, where stock is going up continuously for the last 30 days, it
will forget the experience from day 250 to day 300 where stock had a very low valuation because
of poor financial performance.

**Correlation Problem:** If our agent is learning only from experiences of the past few days then in
the short term the agent can think that Amazon stock will go down, hence will SELL.

But, if you look at the long term view, stock might be a BUY.

As a consequence, it will be more efficient to make use of previous experience, by learning with
it multiple times.

So the solution is to create a “replay buffer.” These stores experience tuples while interacting
with the environment, and then we sample a small batch of tuple to feed our neural network.

Experience replay will help us to handle two things:

● Avoid forgetting previous experiences.

● Reduce correlations between experiences.

## Parameter Tuning

Parameters that were tuned are:

● Exploration vs Exploitation parameter - Epsilon and its decay rate.

● State definition: There are various values that can go in state and I will try with various
inputs like 5 days stock price, volume etc.

● Memory size

● Action Size: What actions can the agent do? I decided to use 5 actions: Buy1, Sell1,
Buy2, Sell2, Do nothing. I also evaluated the model using other actions like Buy1Sell 2
and Buy2Sell1, but found that model performance did not change much with it.

● Reward for various actions: Reward for not going bankrupt and to not go near bankruptcy
were defined.

● Neural network architecture:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○ Number of layers

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○ Layer types (relu, Linear )

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;○ Layer parameters

## Results & Conclusion

The agent was trained over 50 episodes and the corresponding models ep10, ep20…ep50 were
stored in the model's directory.

In the Training section, we load these models and test how good the agent has learned.

Final Architecture of the model is a deep Q-network with following layers in the neural network,
Input to Neural network is states and the output is the Q-Values. Neural network has 3 layers
with 64, 32 and 8 units. Output layer is the Q layer with units=action size (i.e. 5).

Experience replay is used to help the neural network to converge better.

![image](https://user-images.githubusercontent.com/47473472/119597131-146f6e80-bda6-11eb-89dd-8b147a6877bf.png)

We compared the performance of agent w.r.t. benchmark portfolio. Agent has produced more
profits than the benchmark portfolio, for both training run and actual test run.

![image](https://user-images.githubusercontent.com/47473472/119597172-23562100-bda6-11eb-98af-2da7b0abbaa6.png)


Performance of agent w.r.t. the benchmark portfolio is almost 30% better. This performance is
not by chance, the same performance is also seen in case of unseen data (as shown in previous
section). Over there as well, the agent performed 30% more than the benchmark model. This
justifies that the model is robust and gives good performance.

## Appendix

#### Data Pre-Processing

First step in data preprocessing was to convert the Date column to date format.

Next step was to make sure that data for both Apple and Amazon are for the same dates. There
are many days where either Apple or Amazon data are not present. Where data is not present for
a particular day, I had 2 options:

1. We could either populate the data from previous day to the missing day or
2. We could delete the record from other stocks.
 
We chose to take the option-2 i.e. of deleting the record from other stock for missing stock date.
So, for e.g. if there is no data present for Apple stock on the 12th Jan 1998, then I would delete
the corresponding day’s record from Amazon stock. This way the days in Apple and Amazon
stock would be the same set.

Some records from Apple and Amazon Stock data have been deleted to make sure that the data is
of the same size. This is not a big number of records- overall 6 records from 5155 records.

Other than this, data is already clean, prices are adjusted for splits etc.

Then we divided the data into training and test sets, in chronological order.

When I decide on the number of episodes to use (say 100), I will divide the dataset in
chronological order from a start date D. So:

● Training data is from Day D to Day D+1000.

● Then we tested from day D+1001 to D+1500.

## References

1. Figure 3 from Python Lessons (pylessons.com)
2. MACHINE LEARNING FOR TRADING: GORDON RITTER:
https://cims.nyu.edu/~ritter/ritter2017machine.pdf
3. Financial Trading as a Game: A Deep Reinforcement Learning Approach | Papers With
Code
4. Deep Reinforcement Learning with Double Q-learning [1509.06461] Deep
Reinforcement Learning with Double Q-learning (arxiv.org)
5. sachink2010/AutomatedStockTrading-DeepQ-Learning
Algorithmic

