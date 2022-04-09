
from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    # FuncAnimation: A class that exposes a callable interface to animate a matplotlib figure.
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    # https://matplotlib.org/gallery/animation/simple_anim.html#sphx-glr-gallery-animation-simple-anim-py
    #anim.save(path + filename, writer='imagemagick', fps=60)
    anim.save(path + filename.replace('.gif', '.mp4'), writer='ffmpeg', fps=30)

#Make gym env
env = gym.make('BreakoutNoFrameskip-v4')

#Run the env
observation = env.reset()
frames = []
for t in range(10000):
    #Render to frames buffer
    frames.append(env.render(mode="rgb_array"))
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)
    if done:
        break
env.close()
save_frames_as_gif(frames)