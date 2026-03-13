### make the current design more RESTful for system designer/algorithm researcher

1. I want to use direct api calls like POST, PUT, GET, DELETE when executing the model, the process mush be as zero-cost as possible
2. the execution plan or specific module of code should be managed as resource that can be GET or POST, which means patching code will be very easy for the user
3. For each module, please equip with example input and ouput data. For later phase, design unit test on these input and output data. 