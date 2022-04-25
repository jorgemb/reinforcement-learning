FROM jupyter/scipy-notebook:latest as notebook

# Add vim binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions/vim_binding \
    && jupyter nbextension install https://raw.githubusercontent.com/lambdalisue/jupyter-vim-binding/master/vim_binding.js --nbextensions=$(jupyter --data-dir)/nbextensions/vim_binding \
    && jupyter nbextension enable vim_binding/vim_binding

# References
# Vectors: xtensor 
# Plotting: xplot https://github.com/QuantStack/xplot
# Plotting: matplotlib-cpp https://github.com/lava/matplotlib-cpp
# RUN mamba install xtensor fmt root -y

# Using xeus instead of root
RUN mamba install -c conda-forge fmt xeus-cling xtensor -y

# Mount external libraries
VOLUME /rl/

# Copy build information
#ENTRYPOINT [ "start.sh", "root", "--notebook", "--NotebookApp.token", "''"]
ENTRYPOINT [ "start.sh", "jupyter", "notebook", "--NotebookApp.token", "''"]
