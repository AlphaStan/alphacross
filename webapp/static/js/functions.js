function resetBoard(result, status, xhr){
    console.log("Reset board");
    // Reset buttons class
    var buttons = board.getElementsByTagName('button');
    var button;
    for (var k in buttons){
        button = buttons[k];
        try{
            button.classList.remove(...button.classList);
        }
        catch(error){
            console.log('Cannot access element ' + k + ' of buttons: ' + error);
        }
    }
    document.getElementById('msg').innerHTML = '';
    gameIsFinished = false;
}


function defaultError(xhr, status, error) {
    console.log(xhr.status + ": " + xhr.responseText);
}


function handleGridUpdate(result, status, xhr){
    var parsed_result = $.parseJSON(result);
    var buttonClass = (parsed_result["player_id"] == 1) ? "red" : "yellow";
    var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["row_id"]) + ')');
    var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["col_id"]) + ')');
    if (parsed_result["column_is_full"]) {
        document.getElementById('msg').innerHTML = 'Sorry, column ' + (1 + parsed_result["col_id"]) + ' is full';
    }
    else{
        cell.firstElementChild.classList.add(buttonClass);
        document.getElementById('msg').innerHTML = '';
    }
    if (parsed_result["has_won"]) {
        if (activeAI){
            document.getElementById('msg').innerHTML = 'You have won!';
        }
        else{
            document.getElementById('msg').innerHTML = 'Player ' + parsed_result["player_id"] + ' has won!';
        }
        gameIsFinished = true;
    }
    if (activeAI & !parsed_result["column_is_full"] & !parsed_result["has_won"]) {
        var agentButtonClass = (parsed_result["agent_id"] == 2) ? "yellow" : "red";
        var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["agent_row_id"]) + ')');
        var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["agent_col_id"]) + ')');
        cell.firstElementChild.classList.add(agentButtonClass);
        document.getElementById('msg').innerHTML = '';
        if (parsed_result["agent_has_won"]) {
            document.getElementById('msg').innerHTML = 'You have lost!';
            gameIsFinished = true;
        }
    }
}