# jigsaw_robot
final project from Robotics 2020 spring

# Setup Env


## Usage
```
python puzzle_solver.py -i <image_path> -o <reference_image_path>
```
- the result will be save in ```./results/<img_name>/```
- position information that the robot arm needs is save to ```./results/<img_name>/infp.txt```, a text file with json format
- example:
```
{'0': {'posx': 1670, 
       'posy': 2444, 
       'orientation': 270, 
       'targetx': 1201, 
       'targety': 245}, 
 '1': {'posx': 1041, 
       'posy': 2439, 
       'orientation': 270, 
       'targetx': 721, 
       'targety': 1023}, 
 ...
}
```
- if you wand to reload it as a dictionary: 
```
with open('info.txt', 'r') as f:
    info = json.load(f)
```

## Test Remove Background
```
python testRemoveBG.py <image_path>
```
- the results will be save in the folder ```./results/test/```

## Reference
