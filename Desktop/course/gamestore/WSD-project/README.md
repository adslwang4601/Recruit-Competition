Project Plan
-----------------------

### 1. Team

* 661588 Christian Addy
* 599540 Qiang Huang
* 601098 Qianqian Qin


### 2. Goal

In this project, the group aims to develop a online game store for JavaScript games. The users of this web application include players and developers. Games can be added and sold by developers, and then can be purchased and played online by players. The game will be added by game developers using an URL to an HTML file that contains all assets and JavaScript links to JavaScript files.The application will be fully functional and fully tested.

### 3. Application Functionalities

#### 3.1 Features
##### Basic Features
* Authentication (as a player or a developer)
	- Login, Logout, Register, Email validation
	- How to implement: there are some ways to realize authentication. For example, using django.contrib.auth or the package Django-Registration-Redux. For email validation: django.core.mail. An official documentation realated to authentication: [User authentication in Django](https://docs.djangoproject.com/en/1.11/topics/auth/) 

* Basic player functionalities 
	- Buy games
	- Play games only if have purchased them
	- Searching games
	- See a list of purchased games
	- See top scores for each game
	- How to implement: django.contrib.auth can be used (user group: players)

* Basic developer functionalities 
	- Basic game inventory
	-  Manage their own game(s) (add/remove/modify) and set price(s) for the game(s) in their own inventory
	-  See sale statistics
	- How to implement: django.contrib.auth can be used (user group:developers)

* Game/service interaction
	- Messages are exchanged between game and the game service with **window.postMessage**.
	- From the game to the service: Inform a new score (then the score is recorded).
	- From the service to the game: Inform error information.


##### Advanced Features
* Save/load and resolution feature
	- From the game to the service: Save the sent game state, Request a game state previously saved, Settings telling game specific configuration(resolution).
	- From the service to the game: Load a saved game state.

* Own game
	- Design a simple JavaScript game communicating with service.

* Social media sharing 
	- Share games in some social media sites. The shared contents include: a description and an image.
	- How to implement: using the package django-social-share
	
#### 3.2 Development Technologies
* Languages: HTML, CSS, Javascript, Python, SQL
* Web framework: Django
* Javascript library: jQuery 
* CSS library: Bootstrap
* Database system: SQLite/MySQL/PostgreSQL
* Testing: Python built-in "unittest"/django.test
* Deployment: Heroku

#### 3.3 Models  

* Game  
    - publisher - user foreign key  
    - title  
    - description
    - category
    - price
    - image


* Category
    - id
    - name

* Profile
    - user
    - ownedGames

* Score
    - game
    - player 
    - score
    - date

* GameSale
    - buyer
    - game
    - date  

* Messages(used to interact with  game/service) 
    - score
    - save
    - error
    - setting


#### 3.4 Views

* Player related  
_list purchased games, buy/play game, high scores, social media sharing_

* Developer related  
_list developer games, list sale statistics, add/delete/modify game from own inventory_  

* authentication related  
_log in , log out, sign up_

* SavedGame related  
_save game and load game_


#### 3.5 Testing
An official documentation about testing a web application recommends writing tests with the built-in "unittest" module in Python stardard library. But it still involves some other ways such as django.test.runner.DiscoverRunner.

The detailed information and tutorials can be found here:
[Testing in Django](https://docs.djangoproject.com/en/1.11/topics/testing/)

### 3.6 Priorities
The priorities of the project: 1. mandatory requirements   2. advanced features.

First we will implement the authentication requirement for the game. Once authentication is completed, we will add basic player and developer functionalities. After that we can start to look at the game-service interaction. We will also consider adding some features simultaneously. For example, basic player functionalities and payment functionality should be considered almost at the same time. After the mandatory requirements are met, we will try to realize the advanced features we decided to added.

While implementing these different features, we will constantly check our quality of work. The written code will be commented by each member to prevent any miscommunication. In addition testing will take place at the same time as functionalities are implemented.

### 4. Steps and Schedule

#### 4.1 Steps

1. Planning (DL: 21.12.2017 midnight)
2. Implementing the function Authentication
3. Constructing models
4. Hooking up views
5. Making templates
6. Game/service interaction
7. Testing the roughly finished project locally
8. Deploying the finished project to Heroku
9. Final submission (DL: 19.2.2018 midnight)
10. Project demonstration

Some steps are interactive. During the implementation phase (Step 2-6), appropriate testing will be done after finishing each function.

#### 4.2 Schedule
* 16 Dec-21 Dec: Researching the topic, Collecting materials and Making the project plan.
* 22 Dec-9 Jan: Realizing the function Authentication. And design of models is accomplished or towards the end.
* 10 Jan-6 Feb: Developing views, templates and the function Game/service interaction. Trying to do the initial Heroku deployment.
* 7 Feb-18 Feb: Deploying the project first locally, and then to Heroku. Writing the final documentation.

The schedule is roughly designed. The practical and detailed timeline probably changes and will be recorded during the process of the project.

### 5. Practical Arrangement

#### 5.1 Meeting 

* Frequency: Once or twice a week. It will vary based on project progress.
* Location: Otaniemi Campus or online (Skype or Google Hangouts), depending on the situation. The group will try to hold face-to-face meetings regularly.
* Purpose: 
	- Tracking the progress of the project.
	- Sharing information between the team.
	- Proposing suggestions to the project.
	- Making some adjustments if necessary.
	- Distributing tasks of the next week.

#### 5.2 Project Management Tools

* Gitlab: Used for source code management, version control and issue tracking, etc.
* Slack: Team discussion.
* Trello: Task management, including task assignment and task status tracking, etc.
* Google Drive: Sharing relevant materials.

#### 5.3 Communication Tools

* Wechat: Instant communication if necessary, e.g., sending notifications to members about important topics posted to the project management tools.
	
### 6. Risk Analysis

#### Security problems
The security problems include problems with authorization and authentication, the risk of users tampering with data coming back from the payment service or possibility of user doing script injections to items shown to other users, etc.

How to handle:
1. Collecting relevant materials, such as some related worth learning project involving security issues handling.
2. Discussing and analyzing possible security flaws related to this project.
3. Researching possible solutions and trying to implement them.  

#### Failure of deployment at the final phase 
How to handle:
1. Doing the initial Heroku deployment early (when an initial application is built).
2. Deploying the project when debugging during implementation phase.
3. Tracking Heroku resource usage, and notice not to run out of limited free resources each month. 
4. To save the Heroku resources, developing and testing the project first locally