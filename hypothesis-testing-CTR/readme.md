# Hypothesis Testing

One of the core applications of statistics is to test hypothesis. We use different measurements to make statements about our confidence in the results of experiments. Our confidence in the outcome shapes our view of the world. In this example, we're going to test a single hypothesis in a few different ways. In all cases we're going to try and answer questions related to a commonly tracked metric in websites, the ["Click Through Rate" -- CTR](#TODO).

CTR is a very important metric for digital advertising, but also for e-commerce sites, and any other website attempting to convert website users to customers of some kind of product. CTR could be used to measure how many users signup for a particular event on EventBrite; or it could be used to track the likelihood that a particular advertisement gets clicked on from Google's homepage; or it could be used as part of a larger [funnel analysis](#TODO)

Here is the setup for our experiments:

### Webby McWebsite's New Homepage

Your company has a great product called Webby McWebsite. The Webby team has been working hard on a revamp of one of the most important pages on the website. They've asked you to design an AB test for them to decide if they should keep the new design or stick with the old one. Depending on how it goes, the team says they'd really like to move forward with a redesign of the whole website based on the design principles from the redesign.

Can we design an experiment to answer these questions for the web team?

## Aside: Data Generation

In this example we do not have any real data on CTR, instead we're going to model CTR using a binomial distribution and generate data with different parameters `n` and `p` to serve as our sample size. This has some benefits in the learning context -- specifically that we KNOW the hidden distribution that our data should follow. If our experiments determine something that isn't __true__ then we'll know, and have something interesting to think about.

In the real world this data might have been collected in a number of different ways, which may have their own strengths and drawbacks. Lets take a look at `data_generation.py` to see how this is done.

These generation steps will get more complex as we go on, to account for new assumptions and variables, but we start with a straightforward (if naive) series of Bernoulli trials.

## Part One: Experimental Design

We're going to change the narrative about "Webby" many times during this example, to simulate different situations which we may really find ourselves in. These assumptions will range from  "Webby is a __big__ company, we have __everything__ we could ever ask for." to "Webby has very limited resources, do the best with what we've got!"

Experiments in the wild all depend on your individual goals for these experiments. Entertaining both the "happy path" and more challenging situations can be very valuable.

#### Scaffolding Our Test

Although we'd love to find that the new website is __better__ than the old website, we're not even sure that the website itself is the driving force in click through rate. As a result, in this first experiment, we're not limiting ourselves to discovering that the new website is better -- if it's significantly worse that is still a relevant and important finding. So lets setup our first hypothesis to test:

__Null hypothesis__: the new website will not perform any differently compared to the old website.

__Alternate hypothesis__: the new webpage will perform differently.

Great, we know what we want to determine. Soon we're going to have to ask our web-team to gather some data for us. Before we go to the Webby team we need to answer some questions for ourselves.

* What is "strong enough" evidence to keep the new design in favor of the old design?
* What magnitude change in CTR is required before we care?
* What is our tolerance for Type I and Type II error?
* How much data are we going to ask for?

#### Determining Sample Size

Lets answer these questions for 3 scenarios (ultimately we'll write a function that can handle all three as well as any others). We want to know the minimum effect size that interests us, the confidence level we need to have, and the desired power of our experiment. 

###### Big Company, Already Succeeding

Because Webby is experimenting with a critical page, and we're a big company, we want to hold ourselves to a high standard of confidence. We have millions of daily active users, even a 0.1% percent change might mean a change in millions of dollars revenue.

```
```
