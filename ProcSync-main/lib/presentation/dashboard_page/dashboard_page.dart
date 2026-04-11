import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:share_plus/share_plus.dart';

import './bloc/class_creation_bloc.dart';
import './models/class_model.dart';
import '../chat/models/user_model.dart';
import 'package:ProcSync/routes/app_routes.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final List<ClassModel> groups = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadGroups();
  }

  Future<void> _loadGroups() async {
    final snapshot = await FirebaseFirestore.instance
        .collection('groups')
        .where('archived', isEqualTo: false)
        .get();

    final loadedGroups = snapshot.docs
        .map((doc) => ClassModel.fromMap(doc.data(), id: doc.id))
        .toList();

    setState(() {
      groups.clear();
      groups.addAll(loadedGroups);
      isLoading = false;
    });
  }

  void _showCreateGroupDialog(BuildContext context) {
    final nameController = TextEditingController();
    final sectionController = TextEditingController();

    showDialog(
      context: context,
      builder: (_) {
        return BlocProvider.value(
          value: context.read<ClassCreationBloc>(),
          child: AlertDialog(
            title: const Text('Create group'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(hintText: 'Group name'),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: sectionController,
                  decoration: const InputDecoration(hintText: 'Section'),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              BlocConsumer<ClassCreationBloc, ClassCreationState>(
                listener: (context, state) {
                  if (state is ClassCreationSuccess) {
                    setState(() {
                      groups.add(state.classModel);
                    });
                    Navigator.pop(context);
                  } else if (state is ClassCreationFailure) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(state.error)),
                    );
                  }
                },
                builder: (context, state) {
                  return TextButton(
                    onPressed: state is ClassCreationLoading
                        ? null
                        : () {
                            final currentUser =
                                FirebaseAuth.instance.currentUser;
                            if (currentUser != null &&
                                nameController.text.isNotEmpty &&
                                sectionController.text.isNotEmpty) {
                              context.read<ClassCreationBloc>().add(
                                    CreateClassEvent(
                                      name: nameController.text.trim(),
                                      section: sectionController.text.trim(),
                                      teacherUid: currentUser.uid,
                                    ),
                                  );
                            }
                          },
                    child: const Text('Create'),
                  );
                },
              ),
            ],
          ),
        );
      },
    );
  }

  void _editGroupDialog(ClassModel classModel) {
    final nameController = TextEditingController(text: classModel.name);
    final sectionController = TextEditingController(text: classModel.section);

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Edit Group'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(labelText: 'Name'),
            ),
            TextField(
              controller: sectionController,
              decoration: const InputDecoration(labelText: 'Section'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final newName = nameController.text.trim();
              final newSection = sectionController.text.trim();

              if (newName.isEmpty || newSection.isEmpty) return;

              await FirebaseFirestore.instance
                  .collection('groups')
                  .doc(classModel.id)
                  .update({
                'name': newName,
                'section': newSection,
              });

              setState(() {
                final index = groups.indexWhere((g) => g.id == classModel.id);
                if (index != -1) {
                  groups[index] = ClassModel(
                    id: classModel.id,
                    code: classModel.code,
                    name: newName,
                    section: newSection,
                    createdBy: classModel.createdBy,
                    createdAt: classModel.createdAt,
                    archived: classModel.archived,
                  );
                }
              });

              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  void _showProfileOptions() {
    showModalBottomSheet(
      context: context,
      builder: (_) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.logout),
              title: const Text('Switch Account'),
              onTap: () async {
                await FirebaseAuth.instance.signOut();
                Navigator.pop(context);
                // TODO: Redirect to login page here
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildGroupCard(ClassModel classModel) {
    return Card(
      color: Colors.cyan,
      margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      child: ListTile(
        onTap: () {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            final currentUser = UserModel(
              uid: user.uid,
              name: user.displayName ?? 'Teacher',
              email: user.email ?? '',
              role: 'teacher',
            );

            Navigator.pushNamed(
              context,
              AppRoutes.chatPage,
              arguments: {
                'user': currentUser,
                'classModel': classModel,
              },
            );
          }
        },
        title: Text(
          classModel.name,
          style: const TextStyle(color: Colors.white, fontSize: 18),
        ),
        subtitle: Text(
          "Section: ${classModel.section}",
          style: const TextStyle(color: Colors.white70),
        ),
        trailing: PopupMenuButton<String>(
          icon: const Icon(Icons.more_vert, color: Colors.white),
          onSelected: (value) async {
            if (value == 'share') {
              final shareText =
                  'Join my class "${classModel.name}" (Section: ${classModel.section}) using code: ${classModel.code} in ProcSync.';
              Share.share(shareText);
            } else if (value == 'edit') {
              _editGroupDialog(classModel);
            } else if (value == 'archive') {
              await FirebaseFirestore.instance
                  .collection('groups')
                  .doc(classModel.id)
                  .update({'archived': true});

              setState(() {
                groups.removeWhere((g) => g.id == classModel.id);
              });

              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Group archived")),
              );
            }
          },
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'share', child: Text("Share invitation link")),
            PopupMenuItem(value: 'edit', child: Text("Edit")),
            PopupMenuItem(value: 'archive', child: Text("Archive")),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    return BlocProvider(
      create: (_) => ClassCreationBloc(),
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 1,
          leading: IconButton(
            icon: const Icon(Icons.menu, color: Colors.black, size: 30),
            onPressed: () {},
          ),
          centerTitle: true,
          title: const Text(
            'ProcSync',
            style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
          ),
          actions: [
            if (user?.photoURL != null)
              GestureDetector(
                onTap: _showProfileOptions,
                child: CircleAvatar(
                  backgroundImage: NetworkImage(user!.photoURL!),
                  radius: 18,
                ),
              )
            else
              IconButton(
                icon: const Icon(Icons.account_circle, size: 30),
                onPressed: _showProfileOptions,
              ),
            const SizedBox(width: 10),
          ],
        ),
        body: isLoading
            ? const Center(child: CircularProgressIndicator())
            : groups.isEmpty
                ? const Center(child: Text("No groups yet. Create one!"))
                : ListView.builder(
                    itemCount: groups.length,
                    itemBuilder: (context, index) =>
                        _buildGroupCard(groups[index]),
                  ),
        floatingActionButton: FloatingActionButton(
          onPressed: () => _showCreateGroupDialog(context),
          child: const Icon(Icons.add),
        ),
      ),
    );
  }
}
