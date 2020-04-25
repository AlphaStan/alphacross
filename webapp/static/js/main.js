var gameBoard = document.querySelector('#board');
gameBoard.addEventListener('click', getColumnId)

function sendActionToFlask(column_id){
    var urlToPost = '/' + column_id;
    $.ajax({
        url: urlToPost,
        type: 'GET',
        success: function(result, status, xhr){
            console.log("Sent");
            var parsed_result = $.parseJSON(result);
            var agent_id = parsed_result["agent_id"];
            var buttonClass = (agent_id == 1) ? "red" : "black";
            var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["row_id"]) + ')');
            var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["col_id"]) + ')');
            cell.firstElementChild .classList.add(buttonClass);
            console.log("Received");
            },
        error: function(xhr, status, error) {
            console.log(xhr.status + ": " + xhr.responseText);
        }
    });
}

function getColumnId(e){
    if (e.target.tagName !== 'BUTTON') return;

    var targetCell = e.target.parentElement;
    var targetRow = targetCell.parentElement;
    var targetRowCells = [...targetRow.children];

    var column_id = targetRowCells.indexOf(targetCell);

    sendActionToFlask(column_id)
}
