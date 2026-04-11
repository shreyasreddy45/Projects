import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'bloc/upload_bloc.dart';
import 'package:intl/intl.dart';
import '../../widgets/upload_dialog.dart';

class UploadsPage extends StatelessWidget {
  final String userId;
  final String userName;
  final String role;

  const UploadsPage({
    super.key,
    required this.userId,
    required this.userName,
    required this.role,
  });

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => UploadBloc()..add(LoadUploadsEvent(userId, role)),
      child: _UploadsView(
        userId: userId,
        userName: userName,
        role: role,
      ),
    );
  }
}

class _UploadsView extends StatelessWidget {
  final String userId;
  final String userName;
  final String role;

  const _UploadsView({
    required this.userId,
    required this.userName,
    required this.role,
  });

  @override
  Widget build(BuildContext context) {
    final UploadBloc uploadBloc = BlocProvider.of<UploadBloc>(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Uploads')),
      body: BlocBuilder<UploadBloc, UploadState>(
        builder: (context, state) {
          if (state is UploadLoading) {
            return const Center(child: CircularProgressIndicator());
          } else if (state is UploadLoaded) {
            if (state.uploads.isEmpty) {
              return const Center(child: Text('No uploads yet.'));
            }
            return ListView.builder(
              itemCount: state.uploads.length,
              itemBuilder: (_, index) {
                final upload = state.uploads[index];
                return ListTile(
                  leading: const Icon(Icons.insert_drive_file),
                  title: Text(upload['title']),
                  subtitle: Text(
                    'Uploaded on ${DateFormat.yMMMd().format(upload['timestamp'].toDate())}'
                    '${role == "Teacher" ? " by ${upload['uploaderName']}" : ""}',
                  ),
                  onTap: () {
                    // Optional: launch(upload['fileUrl']);
                  },
                );
              },
            );
          } else if (state is UploadError) {
            return Center(child: Text(state.message));
          }
          return const SizedBox.shrink();
        },
      ),
      floatingActionButton: role == 'Student'
          ? FloatingActionButton(
              onPressed: () {
                showDialog(
                  context: context,
                  builder: (_) => BlocProvider.value(
                    value: uploadBloc,
                    child: UploadDialog(
                      userId: userId,
                      userName: userName,
                      role: role,
                    ),
                  ),
                );
              },
              child: const Icon(Icons.upload),
            )
          : null,
    );
  }
}
