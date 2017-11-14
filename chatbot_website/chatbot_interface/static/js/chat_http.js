// Credits goes to https://blog.heroku.com/in_deep_with_django_channels_the_future_of_real_time_apps_in_django

$(function() {

    var chat_zone = $("#chat_zone");
    
    // chatsock.onmessage = function(message) {
    //     var data = JSON.parse(message.data);
    //     chat_zone.prepend(
    //         $("<p class='answer'></p>").text('Bot: ' + data.message)
    //     );
    // };

    $("#chat_form").on("submit", function(event) {

        try {
            var message_elem = $('#message');
            var message_val = message_elem.val();

            if (message_val) {
                // Send the message
                $.getJSON({
                    url: 'chat?question='+message_val,
                    success: function(data){
                        chat_zone.prepend(
                            $("<p class='answer'></p>").text('Bot: ' + data.answer)
                        );
                    }
                });
                message_elem.val('').focus();

                // Add the message to the chat
                chat_zone.prepend(
                    $("<p class='question'></p>").text('You: ' + message_val)
                );
            }
        }
        catch(err) {
            console.error(err.message);
        }

        return false;
    });
});
