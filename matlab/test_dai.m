function test_dai
%TEST_DAI Test the functionality of MEX interface.
  root_dir = fileparts(fileparts(mfilename('fullpath')));
  psi = dai_readfg(fullfile(root_dir, 'tests','alarm.fg'));
  fprintf('Testing JTREE\n');
  [logZ,q,md,qv,qf] = dai(psi, 'JTREE', '[updates=HUGIN,verbose=0]');
  fprintf('Testing BP\n');
  [logZ,q,md,qv,qf] = dai(psi, 'BP', '[updates=SEQMAX,tol=1e-9,maxiter=10000,logdomain=0]');
  fprintf('Finished.\n');
end
