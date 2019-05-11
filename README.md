<h1>Wizard RL Engine</h1>
<h4>Setup</h4>
Activate the virualenv:<br>
<code>$ source Dependencies/bin/activate</code>
<br>
Install dependencies: <br>
<code>$ pip install -r requirements.txt</code>

<h4>Usage</h4>

From the root directory, to run a standard RL game:<br>
<code>$ python3 main.py</code>

To update the requirements.txt file:<br>
<code>$ pip freeze > requirements.txt</code>

#### Nomenclature

* Game
* Round 
* Trick
* Player
* Card
* Deck

<h4>Architechture</h4>

<pre>
Game Engine 
    Card.py <em>Generates the physical cards</em>
    Game.py <em>Manages each round of tricks</em>
    Player.py <em>Manages each player. All agents adopt from this class</em>
    Trick.py 
    Wizard.py


</pre>
    

<h4>Git Commands</h4>
https://github.com/joshnh/Git-Commands

<h4>Style Guide</h4>

https://www.python.org/dev/peps/pep-0008/

https://google.github.io/styleguide/pyguide.html