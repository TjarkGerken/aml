#pip install gymnasium,pygame
import gymnasium as gym
import time

env = gym.make("MountainCarContinuous-v0", render_mode="human")
observation, info = env.reset()
#observation[0] is x pos. 
#observation[1] is velocity (neg values are to the left)
TARGET = 0.45  #target flag is at x pos 0.45
action = [1]  #-1 is left, 1 is right, 0 is stop / no power


def steuer(dt, last_err, target, sensor_rs, integr, Kp, Kd, Ki):
    """

    :param dt: Delta Time since last measurement
    :param last_err: The last result from the Sensor
    :param target: the target variable
    :param sensor_rs: The current sensor result
    :param integr: The integral part of the controller if there were any previous calculations. Else this is 0
    :param Kp: The proportional part of the controller
    :param Kd: The derivative part of the controller
    :param Ki: The integral part of the controller
    :return:
    """
    err = target - sensor_rs
    d_err = err - last_err
    prop = Kp * err
    integr += Ki * err * dt
    deriv = Kd * d_err / dt
    return round(prop, 2), round(integr, 2), deriv, err


last_err = -5000
integr = 0
for i in range(1000):
    #print(f"Step {i} | {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    if i == 0:
        last_err = TARGET - observation[0]

    prop, inte, deriv, err = steuer(1/30, last_err, TARGET, observation[0], integr, 0.55, 0.3, 0.2)
    print(prop + inte + deriv)
    integr = inte
    last_err = err
    if observation[1] < 0:
        action = [-1]
        integr = 0
    else:
        action = [prop + integr + deriv]
    # print(observation)

    env.render()

    if terminated or truncated:
        print(f"flag reached at step {i}. precision: {TARGET - observation[0]} ; {observation[1]}")
        time.sleep(2)
        observation, info = env.reset()

env.close()
