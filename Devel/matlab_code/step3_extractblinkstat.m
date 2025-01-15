function step3_extractblinkstat ()
    data = load('C:\Users\balan\IdeaProjects\pyblinkers\Devel\step3_data_input_extractBlinkStatistic.mat');  % Loads blinkComp, blinkPositions
    
    blinkFits= data.blinkFits; 
    blinkProperties = data.blinkProperties;
    blinks= data.blinks;
    params= data.params;
    blinkStatistics = extractBlinkStatistics(blinks, blinkFits, ...
                                              blinkProperties, params)
    blinkTable = struct2table(blinkStatistics, 'AsArray', true);
    fileName = 'blinkStatistics.xlsx';
    writetable(blinkTable, fileName);
    g=1
end