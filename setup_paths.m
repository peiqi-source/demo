function setup_paths()
    thisFile = mfilename('fullpath');
    rootDir = fileparts(thisFile);

    restoredefaultpath;
    rehash toolboxcache;

    addpath(genpath(fullfile(rootDir, 'experiments')));
    addpath(genpath(fullfile(rootDir, 'utils')));
    addpath(genpath(fullfile(rootDir, 'datasets')));
end