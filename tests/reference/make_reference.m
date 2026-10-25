% Regenerate the EasySpin reference data used by tests/test_reference.py.
% Needs MATLAB with EasySpin 6.0.12 on the path. Run from this folder:
%   matlab -batch "make_reference"
% Each case in cases.json is simulated with the EasySpin function it names, and
% x and y are saved to ref_<case>.mat.

cases = jsondecode(fileread('cases.json'));
names = fieldnames(cases);
for k = 1:numel(names)
  c = cases.(names{k});
  Sys = rowvectors(c.Sys);
  Exp = rowvectors(c.Exp);
  fcn = str2func(c.function);
  [x, y] = fcn(Sys, Exp);
  save(['ref_' names{k} '.mat'], 'x', 'y', '-v7');
  fprintf('%-22s %-8s %d points\n', names{k}, c.function, numel(y));
end

function s = rowvectors(s)
% jsondecode returns numeric vectors as columns; EasySpin examples use rows.
for f = fieldnames(s).'
  v = s.(f{1});
  if isnumeric(v) && isvector(v)
    s.(f{1}) = v(:).';
  end
end
end
