function reset_board(result, status, xhr){
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


function default_error(xhr, status, error) {
    console.log(xhr.status + ": " + xhr.responseText);
}
